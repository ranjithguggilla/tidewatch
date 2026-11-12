"""Loading and cleaning the Beachwatch samples and the daily weather record."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import Config


def load_samples(path: Path, cfg: Config | None = None) -> pd.DataFrame:
    """Read water quality samples, drop unusable rows and collapse duplicate site-days.

    Duplicate site-days (the same site sampled twice on one date) keep the higher count, which
    is the conservative reading for a public-health decision.
    """
    cfg = cfg or Config()
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.dropna(subset=["enterococci_cfu_100ml"]).copy()
    df["enterococci_cfu_100ml"] = df["enterococci_cfu_100ml"].clip(lower=0)

    # Field instruments produce a few impossible values (a 1,040 C water temperature).
    df.loc[~df["water_temperature_c"].between(3, 40), "water_temperature_c"] = np.nan
    df.loc[df["conductivity_ms_cm"] <= 0, "conductivity_ms_cm"] = np.nan

    df = df[df["date"].dt.year >= cfg.first_year]
    df = df.sort_values(["swim_site", "date", "enterococci_cfu_100ml"])
    df = df.drop_duplicates(["swim_site", "date"], keep="last")
    return df.reset_index(drop=True)


def load_weather(path: Path) -> pd.DataFrame:
    """Daily weather for the single Sydney station, reindexed to a gap-free daily calendar."""
    wx = pd.read_csv(path, parse_dates=["date"]).set_index("date").sort_index()
    wx = wx[["max_temp_C", "min_temp_C", "precipitation_mm"]]
    full = pd.date_range(wx.index.min(), wx.index.max(), freq="D")
    wx = wx.reindex(full)
    wx["precipitation_mm"] = wx["precipitation_mm"].fillna(0.0)
    wx[["max_temp_C", "min_temp_C"]] = wx[["max_temp_C", "min_temp_C"]].ffill()
    wx.index.name = "date"
    return wx


def split_by_date(df: pd.DataFrame, cfg: Config) -> dict[str, pd.DataFrame]:
    """Forward-in-time split. Nothing in a later block informs an earlier one."""
    val_start, test_start = pd.Timestamp(cfg.val_start), pd.Timestamp(cfg.test_start)
    return {
        "train": df[df["date"] < val_start],
        "val": df[(df["date"] >= val_start) & (df["date"] < test_start)],
        "test": df[df["date"] >= test_start],
        "dev": df[df["date"] < test_start],
    }
