import numpy as np
import pandas as pd

from tidewatch.evaluate import (
    capture_at,
    capture_knee,
    cluster_bootstrap,
    cluster_se,
    ece,
    metrics,
    one_se_choice,
    per_row_log_loss,
)


def test_metrics_on_perfect_and_random_scores():
    y = np.array([0, 0, 0, 1, 1, 0, 0, 1, 0, 0] * 20)
    perfect = metrics(y, np.where(y == 1, 0.99, 0.01))
    assert perfect["auc"] == 1.0 and perfect["ap"] > 0.99
    rng = np.random.default_rng(0)
    noise = metrics(y, rng.random(len(y)))
    assert 0.35 < noise["auc"] < 0.65


def test_ece_zero_for_calibrated_constant():
    y = np.array([1, 0, 0, 0] * 50)
    assert ece(y, np.full(len(y), 0.25)) < 1e-9


def test_capture_at_top_share():
    y = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    score = np.array([0.9, 0.8, 0.1, 0.2, 0.3, 0.1, 0.1, 0.1, 0.1, 0.1])
    assert capture_at(y, score, 0.2) == 1.0
    assert capture_at(y, -score, 0.2) == 0.0


def test_per_row_log_loss_clips():
    assert np.isfinite(per_row_log_loss(np.array([1.0]), np.array([0.0]))).all()


def test_cluster_se_matches_plain_se_for_singletons():
    rng = np.random.default_rng(1)
    x = rng.normal(size=400)
    plain = x.std(ddof=1) / np.sqrt(len(x))
    assert abs(cluster_se(x, np.arange(len(x))) - plain) / plain < 0.02


def test_one_se_prefers_simpler_model_inside_the_band():
    rng = np.random.default_rng(2)
    clusters = np.repeat(np.arange(40), 25)
    base = rng.normal(0.30, 0.2, 1000)
    losses = {"simple": base + 0.002, "complex": base}
    chosen, info = one_se_choice(losses, clusters, {"simple": 1, "complex": 2})
    assert chosen == "simple" and info["best"] == "complex"
    losses = {"simple": base + 0.5, "complex": base}
    assert one_se_choice(losses, clusters, {"simple": 1, "complex": 2})[0] == "complex"


def test_bootstrap_draws_are_shared_across_models():
    rng = np.random.default_rng(3)
    y = (rng.random(600) < 0.1).astype(float)
    p = np.clip(0.1 + 0.5 * y * rng.random(600), 0.01, 0.99)
    clusters = np.repeat(np.arange(30), 20)
    d = cluster_bootstrap(y, {"a": p, "b": p}, clusters, 30, seed=1)
    assert np.allclose(d["a"]["auc"], d["b"]["auc"])
    assert len(d["a"]["ap"]) > 20


def test_capture_knee_found_on_a_bent_curve():
    rng = np.random.default_rng(4)
    n = 5000
    score = rng.random(n)
    y = (rng.random(n) < np.where(score > 0.9, 0.6, 0.02)).astype(float)
    knee = capture_knee(y, score)
    assert 0.05 < knee["flag_share"] < 0.35
    assert knee["caught"] > 0.4


def test_clusters_from_dates():
    from tidewatch.evaluate import month_clusters

    c = month_clusters(pd.Series(pd.to_datetime(["2020-01-05", "2020-01-20", "2020-02-01"])))
    assert c[0] == c[1] != c[2]
