"""Feature construction. Every feature is computed from information available before the
sample is collected: rain up to the previous day, and lab results at least two days old."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Config
from .neighbours import nearest_sites, site_table

RAIN = ["r1", "r2", "r3", "r7", "r14", "r30", "max3", "days_since_10", "wet_days_7"]
SITE_HISTORY = ["last_log", "last_exc", "gap_days", "n_prev", "exp_rate", "roll20", "roll10_log"]
NEIGHBOURS = ["nb_last_log", "nb_max_log", "nb_dist_km"]
SEASON = ["doy_sin", "doy_cos", "tmax_prev", "tmin_prev"]
CATEGORICAL = ["region"]
GROUPS = {
    "rain": RAIN,
    "site history": SITE_HISTORY,
    "neighbours": NEIGHBOURS,
    "season": SEASON,
    "region": CATEGORICAL,
}
FEATURES = RAIN + SITE_HISTORY + NEIGHBOURS + SEASON + CATEGORICAL
# Columns that exist only for the leakage audit. They are never model inputs.
AUDIT_ONLY = ["rain_sameday", "conductivity", "water_temp"]

PRIOR_RATE = 0.05
PRIOR_WEIGHT = 10.0


def rain_features(wx: pd.DataFrame) -> pd.DataFrame:
    """Rain and temperature summaries indexed by the date they are used on.

    Everything is shifted so that a row dated D only sees days up to D - 1.
    """
    p = wx["precipitation_mm"]
    prior = p.shift(1)
    out = pd.DataFrame(index=wx.index)
    out["r1"] = prior
    out["r2"] = p.shift(2)
    for n in (3, 7, 14, 30):
        out[f"r{n}"] = prior.rolling(n).sum()
    out["max3"] = prior.rolling(3).max()
    out["wet_days_7"] = (prior >= 1).rolling(7).sum()

    pos = np.arange(len(p))
    last_heavy = pd.Series(np.where(prior.to_numpy() >= 10, pos, np.nan), index=p.index).ffill()
    out["days_since_10"] = (pos - last_heavy).fillna(60).clip(upper=60)

    out["tmax_prev"] = wx["max_temp_C"].shift(1)
    out["tmin_prev"] = wx["min_temp_C"].shift(1)
    doy = wx.index.dayofyear.to_numpy()
    out["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    out["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    out["rain_sameday"] = p  # audit only: the daily total is not known at sampling time
    return out


def site_history_features(df: pd.DataFrame, exceed_cfu: float) -> pd.DataFrame:
    """As-of history of each site, built only from that site's earlier samples."""
    d = df.sort_values(["swim_site", "date"]).copy()
    d["y"] = (d["enterococci_cfu_100ml"] > exceed_cfu).astype(float)
    d["logc"] = np.log1p(d["enterococci_cfu_100ml"])
    g = d.groupby("swim_site", sort=False)

    d["last_log"] = g["logc"].shift(1)
    d["last_exc"] = g["y"].shift(1)
    d["gap_days"] = g["date"].diff().dt.days.clip(upper=60)
    d["n_prev"] = g.cumcount()
    prev_sum = g["y"].cumsum() - d["y"]
    d["exp_rate"] = (prev_sum + PRIOR_RATE * PRIOR_WEIGHT) / (d["n_prev"] + PRIOR_WEIGHT)
    d["roll20"] = g["y"].transform(lambda s: s.shift(1).rolling(20, min_periods=3).mean())
    d["roll20"] = d["roll20"].fillna(d["exp_rate"])
    d["roll10_log"] = g["logc"].transform(lambda s: s.shift(1).rolling(10, min_periods=3).mean())
    return d.drop(columns=["logc"])


def neighbour_features(d: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Mean and max log count at the nearest sites, using results at least `result_lag_days` old."""
    tab = site_table(d)
    nbrs = nearest_sites(tab, cfg.n_neighbours)
    d = d.copy()
    d["logc"] = np.log1p(d["enterococci_cfu_100ml"])
    series = {
        s: g[["date", "logc"]]
        .sort_values("date")
        .rename(columns={"date": "nb_date", "logc": "nb_val"})
        for s, g in d.groupby("swim_site")
    }
    lag = pd.Timedelta(days=cfg.result_lag_days)
    pieces = []
    for site, g in d.groupby("swim_site", sort=False):
        left = g[["date"]].copy()
        left["key"] = left["date"] - lag
        left = left.sort_values("key")
        vals, dists = [], []
        for nb, dist in nbrs.get(site, []):
            m = pd.merge_asof(
                left,
                series[nb],
                left_on="key",
                right_on="nb_date",
                tolerance=pd.Timedelta(days=10),
            )
            vals.append(m["nb_val"].to_numpy())
            dists.append(dist)
        frame = pd.DataFrame(np.vstack(vals).T if vals else np.full((len(left), 1), np.nan))
        res = pd.DataFrame(
            {
                "nb_last_log": frame.mean(axis=1).to_numpy(),
                "nb_max_log": frame.max(axis=1).to_numpy(),
                "nb_dist_km": np.mean(dists) if dists else np.nan,
            },
            index=left.index,
        )
        pieces.append(res)
    feats = pd.concat(pieces)
    return d.drop(columns=["logc"]).join(feats)


def build_table(samples: pd.DataFrame, wx: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Join sample rows with all features and the label."""
    d = site_history_features(samples, cfg.exceed_cfu)
    d = neighbour_features(d, cfg)
    rf = rain_features(wx)
    d = d.merge(rf, left_on="date", right_index=True, how="left")
    d = d.dropna(subset=["r30"]).copy()
    d["conductivity"] = d["conductivity_ms_cm"]
    d["water_temp"] = d["water_temperature_c"]
    return d.sort_values(["date", "swim_site"]).reset_index(drop=True)
