import numpy as np
import pytest

from tidewatch import audit, doseresponse, economics
from tidewatch.app import build_app, parse_rain, run_score
from tidewatch.config import Economics
from tidewatch.data import split_by_date
from tidewatch.features import FEATURES, build_table
from tidewatch.models import RainRule, SiteRatePrior, make_boosting, make_logistic, proba
from tidewatch.score import neutral_values, rain_from_daily, score_site_day, site_state


@pytest.fixture(scope="module")
def table(wx, samples, cfg):
    return build_table(samples, wx, cfg)


@pytest.fixture(scope="module")
def splits(table, cfg):
    return split_by_date(table, cfg)


def test_rain_rule_probabilities(table):
    m = RainRule(10, "r3").fit(table[FEATURES], table["y"])
    p = proba(m, table)
    assert set(np.round(np.unique(p), 6)) == {round(m.p_on_, 6), round(m.p_off_, 6)}
    assert m.p_on_ > m.p_off_


def test_site_prior_ignores_weather(table):
    m = SiteRatePrior().fit(table[FEATURES], table["y"])
    shuffled = table.copy()
    shuffled[["r1", "r3", "r7"]] = (
        shuffled[["r1", "r3", "r7"]].sample(frac=1, random_state=0).to_numpy()
    )
    assert np.allclose(proba(m, table), proba(m, shuffled))


def test_models_beat_prior_on_synthetic_rain_signal(splits):
    tr, va = splits["train"], splits["val"]
    from sklearn.metrics import roc_auc_score

    prior = proba(SiteRatePrior().fit(tr[FEATURES], tr["y"]), va)
    for model in (make_logistic(0.1), make_boosting(60, 7)):
        model.fit(tr[FEATURES], tr["y"])
        assert roc_auc_score(va["y"], proba(model, va)) > roc_auc_score(va["y"], prior)


def test_pipelines_survive_unseen_region_and_missing_values(splits):
    tr, va = splits["train"], splits["val"].copy()
    va.loc[va.index[:20], "region"] = "Somewhere New"
    va.loc[va.index[:20], "nb_last_log"] = np.nan
    for model in (make_logistic(0.1), make_boosting(40, 7)):
        model.fit(tr[FEATURES], tr["y"])
        p = proba(model, va)
        assert np.isfinite(p).all() and ((p > 0) & (p < 1)).all()


@pytest.fixture(scope="module")
def bundle(table, splits):
    model = make_logistic(0.1).fit(splits["dev"][FEATURES], splits["dev"]["y"])
    return {
        "model": model,
        "chosen": "logistic",
        "exceed_cfu": 200.0,
        "site_state": site_state(table),
        "neutral": neutral_values(splits["dev"]),
    }


def test_rain_from_daily_matches_definitions():
    f = rain_from_daily([12.0, 0.0, 4.0, 0.0, 0.0, 0.0, 0.0, 30.0])
    assert f["r1"] == 12 and f["r3"] == 16 and f["r7"] == 16 and f["r14"] == 46
    assert f["days_since_10"] == 0 and f["wet_days_7"] == 2 and f["max3"] == 12
    assert rain_from_daily([])["days_since_10"] == 60


def test_scoring_heavy_rain_beats_dry_weather(bundle):
    site = bundle["site_state"].index[0]
    wet = score_site_day(bundle, site, "2020-02-01", [40, 30, 10, 5, 0, 0, 0], 25, 18)
    dry = score_site_day(bundle, site, "2020-02-01", [0] * 14, 25, 18)
    assert 0 < dry["probability"] < wet["probability"] < 1
    assert any(r.group == "rain" and r.delta > 0 for r in wet["reasons"])
    assert wet["level"] in {"Elevated risk", "Swim not recommended"}


def test_last_result_override_changes_the_score(bundle):
    site = bundle["site_state"].index[0]
    a = score_site_day(bundle, site, "2020-02-01", [0] * 7, last_result_cfu=2)["probability"]
    b = score_site_day(bundle, site, "2020-02-01", [0] * 7, last_result_cfu=5000)["probability"]
    # The synthetic samples are independent, so only check that the override is used.
    assert a != b


def test_unknown_site_raises(bundle):
    with pytest.raises(KeyError):
        score_site_day(bundle, "Nowhere Beach", "2020-02-01", [0])


def test_app_helpers_and_smoke(bundle):
    assert parse_rain("12, 0 ;3.5,") == [12.0, 0.0, 3.5]
    site = bundle["site_state"].index[0]
    text = run_score(bundle, site, "2020-02-01", "30, 20, 0", 25.0, 18.0, None)
    assert "Chance this sample exceeds 200" in text
    assert build_app(bundle) is not None


def test_economics_breakeven_and_policies():
    eco = Economics()
    assert 0 < eco.breakeven_probability < 1
    y = np.array([1, 0, 0, 0] * 25)
    assert economics.net_benefit(y, np.zeros(len(y), bool), eco) == 0
    assert economics.net_benefit(y, y.astype(bool), eco) == economics.oracle_net_benefit(y, eco)
    p = np.where(y == 1, 0.9, 0.01)
    assert economics.policy_flags(p, eco).sum() == y.sum()
    k, _ = economics.best_rule_threshold(y, y * 20.0, eco)
    assert k <= 20


def test_economics_sensitivity_rows():
    y = np.array([1, 0, 0, 0] * 25)
    rows = economics.sensitivity(
        y, np.where(y == 1, 0.8, 0.02), y.astype(bool), [1, 5], Economics()
    )
    assert rows[0]["breakeven"] < rows[1]["breakeven"]
    assert rows[0]["never"] == 0


def test_dose_response_finds_a_break(table):
    curve = doseresponse.rate_curve(table.assign(region="x"))
    assert curve["n"].sum() == len(table)
    res = doseresponse.fit_breakpoint(curve)
    assert res["breakpoint_mm"] > 0
    dc = doseresponse.distance_correlations(table, max_rows=800)
    assert all(0 <= v <= 1 for v in dc.values())


def test_rolling_recalibration_moves_toward_observed_rate(splits):
    te = splits["test"]
    p = np.full(len(te), 0.5)
    out = audit.rolling_recalibration(te, p)
    assert out.shape == p.shape
    assert out.mean() <= p.mean()


def test_yearly_calibration_columns(splits):
    te = splits["test"]
    cal = audit.yearly_calibration(te, np.full(len(te), 0.1))
    assert {"year", "observed", "predicted", "n"} <= set(cal.columns)


def test_pipeline_end_to_end_on_synthetic_data(tmp_path, wx, samples):
    from tidewatch.config import Config
    from tidewatch.pipeline import run

    data = tmp_path / "raw"
    data.mkdir()
    samples.assign(date=samples["date"].dt.strftime("%Y-%m-%d")).to_csv(
        data / "water_quality.csv", index=False
    )
    wx_out = wx.reset_index().assign(latitude=0.0, longitude=0.0)
    wx_out["date"] = wx_out["date"].dt.strftime("%Y-%m-%d")
    wx_out.to_csv(data / "weather.csv", index=False)
    cfg = Config(
        val_start="2019-09-01",
        test_start="2020-03-01",
        first_year=2018,
        data_dir=data,
        out_dir=tmp_path / "out",
        asset_dir=tmp_path / "assets",
        n_boot=20,
    )
    out = run(cfg, fast=True)
    assert (tmp_path / "out" / "metrics.json").exists()
    assert (tmp_path / "out" / "model.joblib").exists()
    assert out["selection"]["chosen"] in {"site prior", "rain rule", "logistic", "boosting"}
    assert 0 < out["test"][out["selection"]["chosen"]]["auc"] <= 1
