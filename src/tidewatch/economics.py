"""Turning probabilities into advisory decisions and a dollar figure, under stated assumptions."""

from __future__ import annotations

import numpy as np

from .config import Economics


def net_benefit(y: np.ndarray, flag: np.ndarray, eco: Economics) -> float:
    """Dollars gained per site-day: true alerts avoid illness, false alerts waste swims."""
    y = np.asarray(y, dtype=bool)
    flag = np.asarray(flag, dtype=bool)
    tp = np.sum(flag & y)
    fp = np.sum(flag & ~y)
    return float((tp * eco.benefit_true_alert - fp * eco.cost_false_alert) / len(y))


def policy_flags(p: np.ndarray, eco: Economics) -> np.ndarray:
    """Flag a site-day when the predicted chance exceeds the break-even probability."""
    return np.asarray(p) >= eco.breakeven_probability


def best_rule_threshold(
    y, rain: np.ndarray, eco: Economics, grid=(1, 2, 3, 5, 8, 10, 15, 20, 30, 50)
):
    """Rain-rule threshold with the best net benefit on the data given (validation in practice)."""
    scores = {k: net_benefit(y, rain >= k, eco) for k in grid}
    best = max(scores, key=scores.get)
    return best, scores


def oracle_net_benefit(y, eco: Economics) -> float:
    y = np.asarray(y, dtype=bool)
    return float(y.sum() * eco.benefit_true_alert / len(y))


def sensitivity(y, p, rain_flags, ratios, base: Economics):
    """Net benefit per site-day as the cost of a false alert varies.

    `ratios` are lost-value-per-swim values; everything else stays at the stated assumptions.
    """
    rows = []
    for v in ratios:
        eco = Economics(
            swimmers_per_site_day=base.swimmers_per_site_day,
            illness_risk=base.illness_risk,
            cost_per_illness=base.cost_per_illness,
            compliance=base.compliance,
            lost_value_per_swim=float(v),
        )
        rows.append(
            {
                "lost_value_per_swim": float(v),
                "breakeven": eco.breakeven_probability,
                "model": net_benefit(y, policy_flags(p, eco), eco),
                "rain_rule": net_benefit(y, rain_flags, eco),
                "always": net_benefit(y, np.ones(len(y), bool), eco),
                "never": 0.0,
            }
        )
    return rows
