"""Metrics, month-cluster bootstrap and the one-standard-error model choice."""

from __future__ import annotations

import numpy as np
import pandas as pd
from kneed import KneeLocator
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

EPS = 1e-6


def per_row_log_loss(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    p = np.clip(p, EPS, 1 - EPS)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def ece(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    """Expected calibration error with quantile bins, which suits a rare positive class."""
    order = np.argsort(p)
    chunks = np.array_split(order, bins)
    n = len(y)
    return float(sum(len(c) / n * abs(y[c].mean() - p[c].mean()) for c in chunks if len(c)))


def capture_at(y: np.ndarray, score: np.ndarray, share: float = 0.10) -> float:
    """Share of all exceedances that land in the top `share` of site-days by score."""
    k = max(1, int(round(share * len(y))))
    idx = np.argsort(-score, kind="stable")[:k]
    return float(y[idx].sum() / max(y.sum(), 1))


def metrics(y, p) -> dict[str, float]:
    y = np.asarray(y, dtype=float)
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    return {
        "auc": float(roc_auc_score(y, p)),
        "ap": float(average_precision_score(y, p)),
        "log_loss": float(log_loss(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "ece": ece(y, p),
        "capture10": capture_at(y, p, 0.10),
    }


def month_clusters(dates: pd.Series) -> np.ndarray:
    return (dates.dt.year * 12 + dates.dt.month).to_numpy()


def cluster_bootstrap(
    y, preds: dict[str, np.ndarray], clusters: np.ndarray, n_boot: int, seed: int
):
    """Resample whole months so that neighbouring site-days stay together.

    Returns {model: {metric: array of draws}}. All models share the same draws, which makes
    paired differences meaningful.
    """
    rng = np.random.default_rng(seed)
    y = np.asarray(y, dtype=float)
    uniq = np.unique(clusters)
    members = {c: np.flatnonzero(clusters == c) for c in uniq}
    draws = {m: {k: [] for k in ("auc", "ap", "log_loss", "capture10")} for m in preds}
    for _ in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([members[c] for c in pick])
        yb = y[idx]
        if yb.sum() == 0:
            continue
        for m, p in preds.items():
            pb = np.clip(p[idx], EPS, 1 - EPS)
            draws[m]["auc"].append(roc_auc_score(yb, pb))
            draws[m]["ap"].append(average_precision_score(yb, pb))
            draws[m]["log_loss"].append(log_loss(yb, pb))
            draws[m]["capture10"].append(capture_at(yb, pb, 0.10))
    return {m: {k: np.array(v) for k, v in d.items()} for m, d in draws.items()}


def interval(draws: np.ndarray, level: float = 0.95) -> tuple[float, float]:
    lo, hi = np.quantile(draws, [(1 - level) / 2, 1 - (1 - level) / 2])
    return float(lo), float(hi)


def cluster_se(loss: np.ndarray, clusters: np.ndarray) -> float:
    """Standard error of a mean loss when rows are grouped into clusters."""
    df = pd.DataFrame({"l": loss, "c": clusters})
    g = df.groupby("c")["l"].agg(["sum", "count"])
    mean = loss.mean()
    resid = g["sum"] - mean * g["count"]
    n = len(loss)
    k = len(g)
    return float(np.sqrt(k / (k - 1) * (resid**2).sum()) / n)


def one_se_choice(
    val_losses: dict[str, np.ndarray], clusters: np.ndarray, complexity: dict[str, int]
):
    """Pick the simplest model whose validation loss is within one standard error of the best."""
    means = {m: float(v.mean()) for m, v in val_losses.items()}
    best = min(means, key=means.get)
    se = cluster_se(val_losses[best], clusters)
    eligible = [m for m, v in means.items() if v <= means[best] + se]
    chosen = min(eligible, key=lambda m: complexity[m])
    return chosen, {"means": means, "best": best, "se_of_best": se, "eligible": eligible}


def capture_knee(y, score, grid: int = 200) -> dict[str, float]:
    """Where the capture curve stops paying off, found with kneed.

    Returns the share of site-days flagged at the knee and the share of exceedances caught there.
    """
    y = np.asarray(y, dtype=float)
    order = np.argsort(-np.asarray(score), kind="stable")
    cum = np.cumsum(y[order]) / y.sum()
    xs = np.linspace(0.005, 0.5, grid)
    ys = np.array([cum[max(int(x * len(y)) - 1, 0)] for x in xs])
    knee = KneeLocator(xs, ys, curve="concave", direction="increasing", S=1.0).knee
    if knee is None:
        return {"flag_share": float("nan"), "caught": float("nan")}
    return {"flag_share": float(knee), "caught": float(ys[int(np.argmin(np.abs(xs - knee)))])}
