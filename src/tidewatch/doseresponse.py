"""How exceedance risk responds to rain: a fitted breakpoint and a distance correlation."""

from __future__ import annotations

import dcor
import numpy as np
import pandas as pd
import pwlf

TAIL_PROBS = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.88, 0.94, 0.97, 0.99, 1.0)


def rate_curve(df: pd.DataFrame, rain_col: str = "r3") -> pd.DataFrame:
    """Exceedance rate against rain: one bin for dry days plus quantile bins of the wet days.

    The upper quantiles are packed closer together so the heavy-rain tail gets several points
    instead of one bin that averages a 15 mm day with a 120 mm day.
    """
    dry = df[df[rain_col] <= 0.05]
    wet = df[df[rain_col] > 0.05]
    pieces = [dry.assign(bin=-1)]
    if len(wet):
        q = pd.qcut(wet[rain_col], q=list(TAIL_PROBS), duplicates="drop")
        pieces.append(wet.assign(bin=q.cat.codes))
    both = pd.concat(pieces)
    out = both.groupby("bin").agg(rain=(rain_col, "mean"), rate=("y", "mean"), n=("y", "size"))
    return out.reset_index(drop=True)


def fit_breakpoint(curve: pd.DataFrame) -> dict[str, float]:
    """Two-segment piecewise-linear fit (pwlf) of exceedance rate against recent rain."""
    x = curve["rain"].to_numpy(dtype=float)
    y = curve["rate"].to_numpy(dtype=float)
    w = np.sqrt(curve["n"].to_numpy(dtype=float))
    model = pwlf.PiecewiseLinFit(x, y, weights=w)
    breaks = model.fit(2)
    bp = float(breaks[1])
    slopes = model.calc_slopes()
    return {
        "breakpoint_mm": bp,
        # A break only means something when dry weather is mostly clean. Where 5% of dry-day
        # samples already exceed, rain is not what drives the exceedances.
        "has_threshold": bool(bp >= 2.0 and y[0] < 0.03),
        "slope_below_pp_per_10mm": float(slopes[0] * 10 * 100),
        "slope_above_pp_per_10mm": float(slopes[1] * 10 * 100),
        "dry_rate": float(y[0]),
    }


def regional_dose_response(df: pd.DataFrame, min_rows: int = 3000) -> dict[str, dict]:
    out = {}
    for region, g in df.groupby("region"):
        if len(g) < min_rows:
            continue
        curve = rate_curve(g)
        res = fit_breakpoint(curve)
        res["n"] = int(len(g))
        res["curve"] = curve.to_dict(orient="list")
        out[str(region)] = res
    return out


def distance_correlations(
    df: pd.DataFrame, cols=("r1", "r3", "r7", "r14", "r30"), max_rows: int = 40000, seed: int = 7
) -> dict[str, float]:
    """Distance correlation between log bacteria counts and each rain window.

    Unlike Pearson it also picks up non-monotone dependence, which is why it is worth a look
    next to the fitted dose-response curve.
    """
    d = df.sample(min(len(df), max_rows), random_state=seed)
    target = np.log1p(d["enterococci_cfu_100ml"].to_numpy(dtype=float))
    return {c: float(dcor.distance_correlation(d[c].to_numpy(dtype=float), target)) for c in cols}
