import numpy as np
import pandas as pd
import pytest

from tidewatch.config import Config


def make_wx(n_days: int = 900, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2018-01-01", periods=n_days, freq="D", name="date")
    rain = np.where(rng.random(n_days) < 0.3, rng.gamma(1.2, 8, n_days), 0.0).round(1)
    return pd.DataFrame(
        {
            "max_temp_C": 22 + 5 * rng.standard_normal(n_days),
            "min_temp_C": 14 + 3 * rng.standard_normal(n_days),
            "precipitation_mm": rain,
        },
        index=idx,
    )


def make_samples(wx: pd.DataFrame, n_sites: int = 6, seed: int = 1) -> pd.DataFrame:
    """Synthetic weekly samples whose exceedance risk rises with rain two days before."""
    rng = np.random.default_rng(seed)
    rows = []
    for s in range(n_sites):
        lat, lon = -33.85 + 0.01 * s, 151.2 + 0.01 * s
        for d in wx.index[10::6]:
            recent = (
                wx["precipitation_mm"]
                .loc[d - pd.Timedelta(days=3) : d - pd.Timedelta(days=1)]
                .sum()
            )
            p = 1 / (1 + np.exp(-(-3.2 + 0.07 * recent + 0.15 * s)))
            cfu = rng.gamma(2, 400) if rng.random() < p else rng.gamma(1, 12)
            rows.append(
                {
                    "region": "Sydney Harbour" if s % 2 else "Sydney City",
                    "council": "X",
                    "swim_site": f"Site {s}",
                    "date": d,
                    "time": "09:00:00",
                    "enterococci_cfu_100ml": float(cfu),
                    "water_temperature_c": 20.0,
                    "conductivity_ms_cm": 50000.0,
                    "latitude": lat,
                    "longitude": lon,
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture(scope="session")
def wx():
    return make_wx()


@pytest.fixture(scope="session")
def samples(wx):
    return make_samples(wx)


@pytest.fixture(scope="session")
def cfg():
    return Config(val_start="2019-09-01", test_start="2020-03-01", first_year=2018, n_boot=20)
