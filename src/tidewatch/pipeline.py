"""End-to-end run: load, build features, choose a model, test it, audit it, price it, save it."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from . import audit, doseresponse, economics, plots
from .config import Config, Economics
from .data import load_samples, load_weather, split_by_date
from .evaluate import (
    capture_knee,
    cluster_bootstrap,
    interval,
    metrics,
    month_clusters,
    one_se_choice,
    per_row_log_loss,
)
from .features import FEATURES, GROUPS, build_table
from .models import (
    COMPLEXITY,
    RainRule,
    SiteRatePrior,
    make_boosting,
    make_forest,
    make_logistic,
    proba,
)
from .score import neutral_values, site_state


def log(msg: str) -> None:
    print(f"[tidewatch] {msg}", flush=True)


def tune_families(train, val, cfg: Config, fast: bool):
    """Fit each model family on the training years and keep its best setting on validation."""
    results: dict[str, dict] = {}

    def record(name, model, params):
        p = proba(model, val)
        loss = per_row_log_loss(val["y"].to_numpy(dtype=float), p)
        if name not in results or loss.mean() < results[name]["loss"].mean():
            results[name] = {"model": model, "params": params, "loss": loss, "p": p}

    prior = SiteRatePrior().fit(train[FEATURES], train["y"])
    record("site prior", prior, {})
    for k in (1, 2, 3, 5, 8, 10, 15, 20, 30):
        for col in ("r1", "r3", "r7"):
            m = RainRule(k, col).fit(train[FEATURES], train["y"])
            record("rain rule", m, {"threshold_mm": k, "column": col})
    for c in (0.01, 0.1, 1.0):
        record("logistic", make_logistic(c).fit(train[FEATURES], train["y"]), {"C": c})
    if not fast:
        record("forest", make_forest(cfg.seed).fit(train[FEATURES], train["y"]), {"trees": 150})
    for it in (100, 200, 400):
        for leaves in (7, 15):
            m = make_boosting(it, leaves, cfg.seed).fit(train[FEATURES], train["y"])
            record("boosting", m, {"max_iter": it, "max_leaf_nodes": leaves})
    return results


def refit(name: str, params: dict, frame, cfg: Config):
    X, y = frame[FEATURES], frame["y"]
    if name == "site prior":
        return SiteRatePrior().fit(X, y)
    if name == "rain rule":
        return RainRule(params["threshold_mm"], params["column"]).fit(X, y)
    if name == "logistic":
        return make_logistic(params["C"]).fit(X, y)
    if name == "forest":
        return make_forest(cfg.seed).fit(X, y)
    return make_boosting(params["max_iter"], params["max_leaf_nodes"], cfg.seed).fit(X, y)


def group_importance(model, test, base_ap, base_ll, seed=7, repeats=5):
    """Shuffle one feature group at a time (rows together) and see how much the score drops."""
    rng = np.random.default_rng(seed)
    out = {}
    for g, cols in GROUPS.items():
        aps, lls = [], []
        for _ in range(repeats):
            shuffled = test.copy()
            idx = rng.permutation(len(test))
            shuffled[cols] = test[cols].to_numpy()[idx]
            m = metrics(test["y"], proba(model, shuffled))
            aps.append(m["ap"])
            lls.append(m["log_loss"])
        out[g] = {
            "ap_drop": float(base_ap - np.mean(aps)),
            "log_loss_rise": float(np.mean(lls) - base_ll),
        }
    return out


def run(cfg: Config, fast: bool = False) -> dict:
    t0 = time.time()
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    cfg.asset_dir.mkdir(parents=True, exist_ok=True)
    eco = Economics()

    log("loading and building features")
    samples = load_samples(cfg.data_dir / "water_quality.csv", cfg)
    wx = load_weather(cfg.data_dir / "weather.csv")
    table = build_table(samples, wx, cfg)
    sp = split_by_date(table, cfg)
    train, val, test, dev = sp["train"], sp["val"], sp["test"], sp["dev"]
    summary = {
        "rows": int(len(table)),
        "sites": int(table["swim_site"].nunique()),
        "exceed_cfu": cfg.exceed_cfu,
        "exceedance_rate": float(table["y"].mean()),
        "splits": {
            k: {
                "rows": int(len(v)),
                "rate": float(v["y"].mean()),
                "from": str(v["date"].min().date()),
                "to": str(v["date"].max().date()),
            }
            for k, v in sp.items()
        },
    }
    by_year = table.groupby(table["date"].dt.year)["y"].mean()

    log("choosing a model on the validation years")
    fam = tune_families(train, val, cfg, fast)
    val_clusters = month_clusters(val["date"])
    chosen, sel = one_se_choice({k: v["loss"] for k, v in fam.items()}, val_clusters, COMPLEXITY)
    log(f"one-SE choice: {chosen} (best raw: {sel['best']})")
    selection = {
        "chosen": chosen,
        "best_raw": sel["best"],
        "se_of_best": sel["se_of_best"],
        "eligible": sel["eligible"],
        "validation_log_loss": sel["means"],
        "params": {k: v["params"] for k, v in fam.items()},
    }

    log("refitting on train plus validation and scoring the test years")
    final = {k: refit(k, v["params"], dev, cfg) for k, v in fam.items()}
    test_p = {k: proba(m, test) for k, m in final.items()}
    ytest = test["y"].to_numpy(dtype=float)
    test_clusters = month_clusters(test["date"])
    draws = cluster_bootstrap(
        ytest, test_p, test_clusters, cfg.n_boot if not fast else 40, cfg.seed
    )
    table_rows, comp_rows = {}, []
    for k, p in test_p.items():
        m = metrics(ytest, p)
        for key in ("auc", "ap", "log_loss", "capture10"):
            m[key + "_ci"] = list(interval(draws[k][key]))
        table_rows[k] = m
        comp_rows.append(
            {
                "name": k,
                "chosen": k == chosen,
                "ap": m["ap"],
                "ap_lo": m["ap_ci"][0],
                "ap_hi": m["ap_ci"][1],
                "auc": m["auc"],
                "auc_lo": m["auc_ci"][0],
                "auc_hi": m["auc_ci"][1],
            }
        )
    diffs = {}
    for ref in ("rain rule", "site prior"):
        if chosen == ref:
            continue
        n = min(len(draws[chosen]["ap"]), len(draws[ref]["ap"]))
        d_ll = draws[ref]["log_loss"][:n] - draws[chosen]["log_loss"][:n]
        d_ap = draws[chosen]["ap"][:n] - draws[ref]["ap"][:n]
        diffs[ref] = {
            "log_loss_gain": float(table_rows[ref]["log_loss"] - table_rows[chosen]["log_loss"]),
            "log_loss_gain_ci": list(interval(d_ll)),
            "ap_gain": float(table_rows[chosen]["ap"] - table_rows[ref]["ap"]),
            "ap_gain_ci": list(interval(d_ap)),
        }

    chosen_model, p_test = final[chosen], test_p[chosen]

    log("audits")
    leak = audit.leakage_audit(train, val)
    protocol = audit.split_protocol_audit(dev, val, train, cfg.seed) if not fast else {}
    ycal = audit.yearly_calibration(test, p_test)
    p_recal = audit.rolling_recalibration(test, p_test)
    recal_year = audit.yearly_calibration(test, p_recal)
    recal = {
        "log_loss_before": float(metrics(ytest, p_test)["log_loss"]),
        "log_loss_after": float(metrics(ytest, p_recal)["log_loss"]),
        "ece_before": float(metrics(ytest, p_test)["ece"]),
        "ece_after": float(metrics(ytest, p_recal)["ece"]),
        "mean_pred_before": float(p_test.mean()),
        "mean_pred_after": float(p_recal.mean()),
        "observed": float(ytest.mean()),
    }
    cold = audit.cold_start_split(test, p_test)
    importance = group_importance(
        chosen_model, test, table_rows[chosen]["ap"], table_rows[chosen]["log_loss"]
    )

    knee_val = capture_knee(val["y"].to_numpy(dtype=float), fam[chosen]["p"])
    knee_test = capture_knee(ytest, p_test)

    log("rain dose-response")
    dose = doseresponse.regional_dose_response(dev)
    dcorr = doseresponse.distance_correlations(dev)

    log("economics")
    best_k, k_scores = economics.best_rule_threshold(val["y"].to_numpy(), val["r3"].to_numpy(), eco)
    flags = {
        "advise every day": np.ones(len(test), bool),
        "best rain rule (3-day)": test["r3"].to_numpy() >= best_k,
        "site prior": economics.policy_flags(test_p["site prior"], eco),
        "model": economics.policy_flags(p_test, eco),
    }
    policy = {}
    for name, f in flags.items():
        policy[name] = {
            "flag_share": float(f.mean()),
            "caught": float((f & (ytest > 0)).sum() / ytest.sum()),
            "precision": float((f & (ytest > 0)).sum() / max(f.sum(), 1)),
            "net_per_site_day": economics.net_benefit(ytest, f, eco),
        }
    policy["oracle"] = {"net_per_site_day": economics.oracle_net_benefit(ytest, eco)}
    # Paired month bootstrap of model minus rain rule net benefit.
    rng = np.random.default_rng(cfg.seed)
    uniq = np.unique(test_clusters)
    members = {c: np.flatnonzero(test_clusters == c) for c in uniq}
    diff_draws = []
    for _ in range(cfg.n_boot if not fast else 40):
        idx = np.concatenate([members[c] for c in rng.choice(uniq, len(uniq))])
        a = economics.net_benefit(ytest[idx], flags["model"][idx], eco)
        b = economics.net_benefit(ytest[idx], flags["best rain rule (3-day)"][idx], eco)
        diff_draws.append(a - b)
    policy["model_minus_rule_ci"] = list(interval(np.array(diff_draws)))
    sens = economics.sensitivity(
        ytest, p_test, test["r3"].to_numpy() >= best_k, np.linspace(0.5, 12, 24), eco
    )

    log("charts")
    a = cfg.asset_dir
    test_year = int(pd.Timestamp(cfg.test_start).year)
    plots.yearly_rate(
        by_year[by_year.index >= cfg.first_year + 1], test_year, a / "exceedance_by_year.png"
    )
    if dose:
        plots.dose_response(dose, a / "rain_dose_response.png")
    plots.model_comparison(comp_rows, a / "model_comparison.png")
    curves = {}
    for label, p in (
        ("model", p_test),
        ("rain rule", test_p["rain rule"]),
        ("site prior", test_p["site prior"]),
    ):
        order = np.argsort(-p, kind="stable")
        curves[label] = (np.arange(1, len(p) + 1) / len(p), np.cumsum(ytest[order]) / ytest.sum())
    plots.capture_curve(curves, a / "capture_curve.png", knee_val.get("flag_share"))
    plots.leakage_bars(leak, a / "leakage_audit.png")
    if protocol:
        plots.protocol_bars(protocol, a / "validation_designs.png")
    plots.net_benefit_chart(sens, eco.lost_value_per_swim, a / "net_benefit.png")
    plots.calibration_panel(ycal, recal_year, a / "drift_calibration.png")

    metrics_out = {
        "data": summary,
        "selection": selection,
        "test": table_rows,
        "chosen_vs_baselines": diffs,
        "leakage_audit": leak,
        "validation_designs": protocol,
        "drift": {"yearly": ycal.to_dict(orient="list"), "recalibration": recal},
        "cold_start": cold,
        "capture_knee": {"validation": knee_val, "test": knee_test},
        "group_importance": importance,
        "dose_response": {
            k: {kk: vv for kk, vv in v.items() if kk != "curve"} for k, v in dose.items()
        },
        "distance_correlation": dcorr,
        "economics": {
            "assumptions": {k: v for k, v in asdict(eco).items() if k != "extra"},
            "breakeven_probability": eco.breakeven_probability,
            "rule_threshold_mm": best_k,
            "policy": policy,
            "sensitivity": sens,
        },
        "runtime_seconds": round(time.time() - t0, 1),
    }
    (cfg.out_dir / "metrics.json").write_text(json.dumps(metrics_out, indent=2, default=float))

    log("saving model bundle")
    everything = table
    deploy = refit(chosen, fam[chosen]["params"], everything, cfg)
    bundle = {
        "model": deploy,
        "chosen": chosen,
        "features": FEATURES,
        "exceed_cfu": cfg.exceed_cfu,
        "site_state": site_state(everything),
        "neutral": neutral_values(dev),
        "last_sample_date": str(everything["date"].max().date()),
    }
    joblib.dump(bundle, cfg.out_dir / "model.joblib")
    log(f"done in {time.time() - t0:.0f}s")
    return metrics_out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Train and evaluate the beach advisory model.")
    ap.add_argument("--data-dir", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--asset-dir", type=Path, default=None)
    ap.add_argument("--fast", action="store_true", help="skip the forest and the slow audits")
    args = ap.parse_args(argv)
    base = Config()
    cfg = Config(
        data_dir=args.data_dir or base.data_dir,
        out_dir=args.out_dir or base.out_dir,
        asset_dir=args.asset_dir or base.asset_dir,
    )
    run(cfg, fast=args.fast)


if __name__ == "__main__":
    main()
