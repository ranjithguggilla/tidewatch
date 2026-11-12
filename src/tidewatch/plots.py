"""README charts. One quiet palette, direct labels where it helps, nothing decorative."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

BLUE, ORANGE, GREY, RED, GREEN = "#1f5f8b", "#d9822b", "#8a8f98", "#b23a48", "#3b7d4f"
plt.rcParams.update(
    {
        "figure.dpi": 150,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
    }
)


def _save(fig, path: Path):
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def yearly_rate(by_year, test_start_year: int, path: Path):
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.plot(by_year.index, by_year.values * 100, color=BLUE, marker="o", ms=3.5)
    ax.axvspan(test_start_year - 0.5, by_year.index.max() + 0.5, color=ORANGE, alpha=0.12)
    ax.text(test_start_year, by_year.max() * 100 * 0.93, "test years", color=ORANGE)
    ax.set_ylabel("samples above 200 per 100 mL (%)")
    ax.set_title("Exceedance rate by year, all sites")
    _save(fig, path)


def dose_response(results: dict, path: Path):
    regions = list(results)
    fig, axes = plt.subplots(1, len(regions), figsize=(3.4 * len(regions), 3.4), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, r in zip(axes, regions, strict=True):
        c = results[r]["curve"]
        ax.scatter(
            c["rain"], np.array(c["rate"]) * 100, s=np.sqrt(c["n"]) * 1.6, color=BLUE, alpha=0.75
        )
        if results[r]["has_threshold"]:
            ax.axvline(results[r]["breakpoint_mm"], color=ORANGE, lw=1.6)
            ax.set_title(f"{r}\nslope changes at {results[r]['breakpoint_mm']:.0f} mm")
        else:
            ax.set_title(f"{r}\nhigh dry-weather baseline")
        ax.set_xlabel("rain in the previous 3 days (mm)")
    axes[0].set_ylabel("exceedance rate (%)")
    _save(fig, path)


def model_comparison(rows: list[dict], path: Path):
    """rows: name, ap, ap_lo, ap_hi, auc, auc_lo, auc_hi."""
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True)
    y = np.arange(len(rows))[::-1]
    for ax, key, title in zip(axes, ("ap", "auc"), ("Average precision", "ROC AUC"), strict=True):
        for yi, r in zip(y, rows, strict=True):
            ax.errorbar(
                r[key],
                yi,
                xerr=[[r[key] - r[key + "_lo"]], [r[key + "_hi"] - r[key]]],
                fmt="o",
                color=ORANGE if r.get("chosen") else BLUE,
                capsize=3,
            )
        ax.set_title(title)
        ax.set_yticks(y, [r["name"] for r in rows])
    _save(fig, path)


def capture_curve(curves: dict[str, tuple], path: Path, knee_share: float | None = None):
    fig, ax = plt.subplots(figsize=(5.4, 4.2))
    colours = {"model": ORANGE, "rain rule": BLUE, "site prior": GREY}
    for name, (x, y) in curves.items():
        ax.plot(x * 100, y * 100, label=name, color=colours.get(name, GREEN), lw=2)
    ax.plot([0, 100], [0, 100], ls="--", color="#bbbbbb", label="random")
    if knee_share is not None and knee_share == knee_share:
        ax.axvline(knee_share * 100, color=ORANGE, ls=":", lw=1.2)
        ax.text(knee_share * 100 + 1, 8, "knee of the\nvalidation curve", color=ORANGE, fontsize=8)
    ax.set_xlabel("site-days flagged (%)")
    ax.set_ylabel("exceedances caught (%)")
    ax.set_title("Capture curve on the test years")
    ax.legend(frameon=False)
    _save(fig, path)


def leakage_bars(audit: dict, path: Path):
    names = list(audit)
    ap = [audit[n]["ap"] for n in names]
    fig, ax = plt.subplots(figsize=(7.4, 3.2))
    cols = [BLUE if i == 0 else RED for i in range(len(names))]
    ax.barh(names[::-1], ap[::-1], color=cols[::-1])
    for i, v in enumerate(ap[::-1]):
        ax.text(v + 0.003, i, f"{v:.3f}", va="center")
    ax.set_xlabel("validation average precision")
    ax.set_title("Columns that look useful but are not known in advance")
    _save(fig, path)


def protocol_bars(rows: dict, path: Path):
    names = list(rows)
    ap = [rows[n]["auc"] for n in names]
    fig, ax = plt.subplots(figsize=(6.8, 3.0))
    ax.barh(names[::-1], ap[::-1], color=[BLUE, GREY, ORANGE][: len(names)][::-1])
    for i, v in enumerate(ap[::-1]):
        ax.text(v + 0.003, i, f"{v:.3f}", va="center")
    ax.set_xlim(0.8, 0.92)
    ax.set_xlabel("ROC AUC")
    ax.set_title("Same model, three validation designs")
    _save(fig, path)


def net_benefit_chart(rows: list[dict], default_value: float, path: Path):
    x = [r["lost_value_per_swim"] for r in rows]
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.plot(x, [r["model"] for r in rows], color=ORANGE, lw=2, label="model at break-even")
    ax.plot(x, [r["rain_rule"] for r in rows], color=BLUE, lw=2, label="best rain rule")
    ax.axhline(0, color="black", lw=0.8)
    ax.axvline(default_value, color="#999999", ls=":")
    ax.set_xlabel("dollars of enjoyment lost per deterred swim")
    ax.set_ylabel("net benefit per site-day ($)")
    ax.set_title(
        "Advisory value as the cost of a false alert changes\n"
        "(advising every day is far below the axis)",
        fontsize=10,
    )
    ax.legend(frameon=False)
    _save(fig, path)


def calibration_panel(yearly, recal_yearly, path: Path):
    fig, ax = plt.subplots(figsize=(7.4, 3.6))
    ax.plot(yearly["year"], yearly["observed"] * 100, color="black", marker="o", label="observed")
    ax.plot(yearly["year"], yearly["predicted"] * 100, color=ORANGE, marker="s", label="model")
    ax.plot(
        recal_yearly["year"],
        recal_yearly["predicted"] * 100,
        color=BLUE,
        marker="^",
        label="model with rolling shift",
    )
    ax.set_ylabel("mean exceedance rate (%)")
    ax.set_title("Predicted against observed rate in the test years")
    ax.set_xticks(yearly["year"])
    ax.legend(frameon=False)
    _save(fig, path)
