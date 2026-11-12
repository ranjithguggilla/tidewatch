"""Checks on the validation protocol itself: leakage, split choice, drift and cold starts."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from sklearn.model_selection import GroupKFold, KFold

from .evaluate import metrics
from .features import FEATURES
from .models import make_boosting, proba


def _fit_eval(train, test, extra=()):
    m = make_boosting(extra=tuple(extra))
    m.fit(train[FEATURES + list(extra)], train["y"])
    return metrics(test["y"], proba(m, test, tuple(extra)))


def leakage_audit(train: pd.DataFrame, val: pd.DataFrame) -> dict[str, dict]:
    """Add columns that look useful but are not known when the advisory is issued."""
    out = {"as deployed": _fit_eval(train, val)}
    out["+ same-day rain total"] = _fit_eval(train, val, ["rain_sameday"])
    out["+ water temperature"] = _fit_eval(train, val, ["water_temp"])
    out["+ conductivity"] = _fit_eval(train, val, ["conductivity"])
    return out


def split_protocol_audit(dev: pd.DataFrame, val: pd.DataFrame, train: pd.DataFrame, seed: int = 7):
    """The same boosting model scored under three validation designs.

    Random folds let the model train on the days right around the ones it is scored on.
    Site-held-out folds ask about unseen sites. Forward in time is the question the council
    actually has.
    """
    rng_dev = dev.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    rows = {}
    scores = []
    for tr, te in KFold(5, shuffle=True, random_state=seed).split(rng_dev):
        scores.append(_fit_eval(rng_dev.iloc[tr], rng_dev.iloc[te]))
    rows["random 5-fold"] = {k: float(np.mean([s[k] for s in scores])) for k in scores[0]}
    scores = []
    for tr, te in GroupKFold(5).split(dev, groups=dev["swim_site"]):
        scores.append(_fit_eval(dev.iloc[tr], dev.iloc[te]))
    rows["site-held-out 5-fold"] = {k: float(np.mean([s[k] for s in scores])) for k in scores[0]}
    rows["forward in time"] = _fit_eval(train, val)
    return rows


def _logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def _offset_for(y, p) -> float:
    """Logit shift that makes the mean prediction equal the observed rate."""
    z = _logit(p)
    target = float(np.mean(y))

    def f(delta):
        return float(np.mean(1 / (1 + np.exp(-(z + delta)))) - target)

    try:
        return float(brentq(f, -4, 4))
    except ValueError:
        return 0.0


def rolling_recalibration(test: pd.DataFrame, p: np.ndarray, window_days: int = 365) -> np.ndarray:
    """Month by month, shift the logit so the last `window_days` of predictions match what happened.

    The shift for a month uses only outcomes dated before it starts, less the two-day lab lag.
    """
    out = np.asarray(p, dtype=float).copy()
    dates = test["date"].to_numpy()
    y = test["y"].to_numpy(dtype=float)
    month = test["date"].dt.to_period("M")
    for m in month.unique():
        start = m.to_timestamp()
        known = (dates < np.datetime64(start - pd.Timedelta(days=2))) & (
            dates >= np.datetime64(start - pd.Timedelta(days=window_days))
        )
        if known.sum() < 500:
            continue
        delta = _offset_for(y[known], p[known])
        now = (month == m).to_numpy()
        out[now] = 1 / (1 + np.exp(-(_logit(p[now]) + delta)))
    return out


def yearly_calibration(df: pd.DataFrame, p: np.ndarray) -> pd.DataFrame:
    t = pd.DataFrame({"year": df["date"].dt.year.to_numpy(), "y": df["y"].to_numpy(), "p": p})
    return (
        t.groupby("year")
        .agg(observed=("y", "mean"), predicted=("p", "mean"), n=("y", "size"))
        .reset_index()
    )


def cold_start_split(test: pd.DataFrame, p: np.ndarray, min_history: int = 20) -> dict[str, dict]:
    young = (test["n_prev"] < min_history).to_numpy()
    out = {}
    for name, mask in (("established sites", ~young), ("new sites (under 20 samples)", young)):
        if mask.sum() >= 150 and test["y"].to_numpy()[mask].sum() >= 5:
            out[name] = {"n": int(mask.sum()), **metrics(test["y"].to_numpy()[mask], p[mask])}
    return out
