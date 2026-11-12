"""Scoring one site-day from raw inputs, with plain-language reasons. The app calls this and
nothing else, so it is the single place where the serving path is tested."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import Economics
from .features import CATEGORICAL, FEATURES, GROUPS, NEIGHBOURS, RAIN, SEASON, SITE_HISTORY
from .models import proba

DRY_RAIN = {
    "r1": 0.0,
    "r2": 0.0,
    "r3": 0.0,
    "r7": 0.0,
    "r14": 0.0,
    "r30": 0.0,
    "max3": 0.0,
    "days_since_10": 60.0,
    "wet_days_7": 0.0,
}


def rain_from_daily(daily_mm: list[float]) -> dict[str, float]:
    """Rain features from daily totals, most recent day first (index 0 is yesterday).

    Missing days beyond the list are treated as dry, which understates the 30-day total when the
    caller supplies fewer than 30 values.
    """
    a = np.zeros(60)
    vals = np.asarray(daily_mm[:60], dtype=float)
    a[: len(vals)] = np.clip(vals, 0, None)
    heavy = np.flatnonzero(a >= 10)
    return {
        "r1": a[0],
        "r2": a[1],
        "r3": a[:3].sum(),
        "r7": a[:7].sum(),
        "r14": a[:14].sum(),
        "r30": a[:30].sum(),
        "max3": a[:3].max(),
        "days_since_10": float(heavy[0]) if len(heavy) else 60.0,
        "wet_days_7": float((a[:7] >= 1).sum()),
    }


def season_from_date(date: pd.Timestamp, tmax: float, tmin: float) -> dict[str, float]:
    doy = date.dayofyear
    return {
        "doy_sin": float(np.sin(2 * np.pi * doy / 365.25)),
        "doy_cos": float(np.cos(2 * np.pi * doy / 365.25)),
        "tmax_prev": tmax,
        "tmin_prev": tmin,
    }


def site_state(d: pd.DataFrame, n_neighbours_cols=NEIGHBOURS) -> pd.DataFrame:
    """Features describing each site as it stands after its latest sample."""
    rows = {}
    for site, g in d.sort_values("date").groupby("swim_site"):
        logc = np.log1p(g["enterococci_cfu_100ml"].to_numpy())
        y = g["y"].to_numpy()
        n = len(g)
        rows[site] = {
            "region": g["region"].iloc[-1],
            "last_log": float(logc[-1]),
            "last_exc": float(y[-1]),
            "gap_days": 6.0,
            "n_prev": float(n),
            "exp_rate": float((y.sum() + 0.5) / (n + 10.0)),
            "roll20": float(y[-20:].mean()),
            "roll10_log": float(logc[-10:].mean()),
            **{
                c: float(g[c].iloc[-1]) if pd.notna(g[c].iloc[-1]) else np.nan
                for c in n_neighbours_cols
            },
        }
    return pd.DataFrame(rows).T


@dataclass
class Reason:
    group: str
    delta: float
    text: str


def build_row(
    bundle: dict,
    site: str,
    date,
    daily_rain: list[float],
    tmax: float,
    tmin: float,
    last_result_cfu: float | None = None,
) -> pd.DataFrame:
    state = bundle["site_state"]
    if site not in state.index:
        raise KeyError(f"unknown site: {site}")
    base = state.loc[site].to_dict()
    if last_result_cfu is not None:
        lc = float(np.log1p(max(last_result_cfu, 0.0)))
        base["last_log"] = lc
        base["last_exc"] = float(last_result_cfu > bundle["exceed_cfu"])
    row = {
        **base,
        **rain_from_daily(daily_rain),
        **season_from_date(pd.Timestamp(date), tmax, tmin),
    }
    frame = pd.DataFrame([row])
    for c in FEATURES:
        if c not in CATEGORICAL:
            frame[c] = pd.to_numeric(frame[c], errors="coerce")
    return frame[FEATURES]


def advisory_level(p: float, eco: Economics) -> str:
    if p >= eco.breakeven_probability:
        return "Swim not recommended"
    if p >= 0.10:
        return "Elevated risk"
    return "Normal conditions"


def _neutral(bundle: dict, group: str, row: pd.DataFrame) -> pd.DataFrame:
    alt = row.copy()
    med = bundle["neutral"]
    if group == "rain":
        for c, v in DRY_RAIN.items():
            alt[c] = v
    elif group in ("site history", "neighbours", "season"):
        for c in GROUPS[group]:
            alt[c] = med[c]
    return alt


def explain(bundle: dict, row: pd.DataFrame, p: float, top: int = 3) -> list[Reason]:
    """Replace one feature group at a time with a neutral value and see how far the score moves."""
    model = bundle["model"]
    names = {
        "rain": "Recent rain",
        "site history": "This site's own record",
        "neighbours": "Readings at nearby sites",
        "season": "Time of year and temperature",
    }
    out: list[Reason] = []
    for g in ("rain", "site history", "neighbours", "season"):
        p_alt = float(proba(model, _neutral(bundle, g, row))[0])
        delta = p - p_alt
        if abs(delta) < 0.005:
            continue
        direction = "raises" if delta > 0 else "lowers"
        out.append(
            Reason(g, delta, f"{names[g]} {direction} the chance by {abs(delta) * 100:.1f} points")
        )
    out.sort(key=lambda r: -abs(r.delta))
    return out[:top]


def score_site_day(
    bundle: dict,
    site: str,
    date,
    daily_rain: list[float],
    tmax: float = 24.0,
    tmin: float = 16.0,
    last_result_cfu: float | None = None,
) -> dict:
    """Probability that the sample exceeds the threshold, an advisory level and reasons."""
    row = build_row(bundle, site, date, daily_rain, tmax, tmin, last_result_cfu)
    p = float(proba(bundle["model"], row)[0])
    eco = Economics()
    return {
        "site": site,
        "probability": p,
        "level": advisory_level(p, eco),
        "site_baseline": float(bundle["site_state"].loc[site, "exp_rate"]),
        "reasons": explain(bundle, row, p),
        "features": row.iloc[0].to_dict(),
    }


def neutral_values(dev: pd.DataFrame) -> dict[str, float]:
    cols = SITE_HISTORY + NEIGHBOURS + SEASON + RAIN
    return {c: float(dev[c].median()) for c in cols}
