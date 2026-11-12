import numpy as np
import pandas as pd

from tidewatch.config import Config
from tidewatch.data import load_samples, load_weather, split_by_date


def test_load_samples_cleans_and_dedups(tmp_path):
    df = pd.DataFrame(
        {
            "region": ["A"] * 5,
            "council": ["c"] * 5,
            "swim_site": ["s1", "s1", "s1", "s2", "s2"],
            "date": ["2010-01-01", "2010-01-01", "2010-01-08", "2010-01-01", "1991-05-01"],
            "time": ["09:00:00"] * 5,
            "enterococci_cfu_100ml": [10, 500, np.nan, 3, 4],
            "water_temperature_c": [20, 1040, 20, 20, 20],
            "conductivity_ms_cm": [50000, 0, 50000, 50000, 50000],
            "latitude": [-33.8] * 5,
            "longitude": [151.2] * 5,
        }
    )
    p = tmp_path / "w.csv"
    df.to_csv(p, index=False)
    out = load_samples(p, Config())
    # NaN result dropped, 1991 row dropped, duplicate site-day keeps the higher count
    assert len(out) == 2
    assert out.loc[out["swim_site"] == "s1", "enterococci_cfu_100ml"].iloc[0] == 500
    assert out["water_temperature_c"].isna().any()
    assert out["conductivity_ms_cm"].isna().any()


def test_weather_is_gap_free(tmp_path):
    p = tmp_path / "wx.csv"
    pd.DataFrame(
        {
            "date": ["2020-01-01", "2020-01-03"],
            "max_temp_C": [25, 27],
            "min_temp_C": [15, 17],
            "precipitation_mm": [2.0, np.nan],
            "latitude": [0, 0],
            "longitude": [0, 0],
        }
    ).to_csv(p, index=False)
    wx = load_weather(p)
    assert len(wx) == 3
    assert wx["precipitation_mm"].notna().all()
    assert wx.loc["2020-01-02", "max_temp_C"] == 25


def test_split_is_forward_in_time(samples, cfg):
    d = samples.assign(y=0)
    sp = split_by_date(d, cfg)
    assert sp["train"]["date"].max() < sp["val"]["date"].min()
    assert sp["val"]["date"].max() < sp["test"]["date"].min()
    assert len(sp["dev"]) == len(sp["train"]) + len(sp["val"])
