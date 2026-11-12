import numpy as np
import pandas as pd

from tidewatch.features import FEATURES, build_table, rain_features, site_history_features
from tidewatch.neighbours import nearest_sites, site_table


def test_rain_features_only_see_the_past(wx):
    rf = rain_features(wx)
    day = wx.index[100]
    p = wx["precipitation_mm"]
    assert rf.loc[day, "r1"] == p.loc[day - pd.Timedelta(days=1)]
    assert np.isclose(
        rf.loc[day, "r3"], p.loc[day - pd.Timedelta(days=3) : day - pd.Timedelta(days=1)].sum()
    )
    assert rf.loc[day, "rain_sameday"] == p.loc[day]


def test_changing_today_does_not_change_todays_features(wx):
    day = wx.index[200]
    before = rain_features(wx).loc[day, ["r1", "r3", "r7", "r30", "max3", "days_since_10"]]
    wx2 = wx.copy()
    wx2.loc[day, "precipitation_mm"] += 500
    after = rain_features(wx2).loc[day, ["r1", "r3", "r7", "r30", "max3", "days_since_10"]]
    pd.testing.assert_series_equal(before, after)


def test_days_since_heavy_rain():
    idx = pd.date_range("2020-01-01", periods=10, freq="D")
    wx = pd.DataFrame(
        {
            "max_temp_C": 20.0,
            "min_temp_C": 10.0,
            "precipitation_mm": [0, 0, 15, 0, 0, 0, 0, 0, 0, 0.0],
        },
        index=idx,
    )
    rf = rain_features(wx)
    assert rf["days_since_10"].iloc[3] == 0
    assert rf["days_since_10"].iloc[6] == 3
    assert rf["days_since_10"].iloc[1] == 60


def test_site_history_excludes_current_row(samples):
    d = site_history_features(samples, 200.0).sort_values(["swim_site", "date"])
    first = d.groupby("swim_site").head(1)
    assert first["last_log"].isna().all()
    assert (first["n_prev"] == 0).all()
    one = d[d["swim_site"] == "Site 0"].reset_index(drop=True)
    assert np.isclose(one.loc[5, "last_log"], np.log1p(one.loc[4, "enterococci_cfu_100ml"]))
    expected = (one.loc[:4, "y"].sum() + 0.5) / (5 + 10)
    assert np.isclose(one.loc[5, "exp_rate"], expected)


def test_nearest_sites_are_sorted_and_exclude_self(samples):
    tab = site_table(samples)
    nb = nearest_sites(tab, 3)
    for site, lst in nb.items():
        assert site not in [n for n, _ in lst]
        assert [d for _, d in lst] == sorted(d for _, d in lst)
        assert len(lst) == 3


def test_neighbour_features_respect_the_lab_lag(wx, samples, cfg):
    table = build_table(samples, wx, cfg)
    assert set(FEATURES) <= set(table.columns)
    # Rewriting the newest results anywhere must not change features of earlier rows.
    late = samples[samples["date"] > samples["date"].quantile(0.8)].index
    s2 = samples.copy()
    s2.loc[late, "enterococci_cfu_100ml"] = 99999.0
    t2 = build_table(s2, wx, cfg)
    cutoff = samples["date"].quantile(0.8) - pd.Timedelta(days=cfg.result_lag_days)
    a = (
        table[table["date"] <= cutoff]
        .set_index(["swim_site", "date"])[FEATURES]
        .drop(columns="region")
    )
    b = t2[t2["date"] <= cutoff].set_index(["swim_site", "date"])[FEATURES].drop(columns="region")
    pd.testing.assert_frame_equal(a, b)
