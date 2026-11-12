"""Settings shared by the pipeline, the tests and the app."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Config:
    # A sample counts as an exceedance above this many enterococci per 100 mL.
    exceed_cfu: float = 200.0
    # Forward-in-time split: train before val_start, validate until test_start, test after.
    val_start: str = "2017-01-01"
    test_start: str = "2021-01-01"
    first_year: int = 1993
    # Lab results take a day or two, so neighbour and own-site history is lagged.
    result_lag_days: int = 2
    n_neighbours: int = 3
    seed: int = 7
    n_boot: int = 300
    data_dir: Path = ROOT / "data" / "raw"
    out_dir: Path = ROOT / "outputs"
    asset_dir: Path = ROOT / "assets"


@dataclass(frozen=True)
class Economics:
    """Stated assumptions for the advisory cost model. All of them are inputs, not findings."""

    swimmers_per_site_day: float = 200.0
    illness_risk: float = 0.03  # chance a swimmer gets ill on an exceedance day
    cost_per_illness: float = 250.0  # dollars, healthcare plus lost time
    compliance: float = 0.5  # share of swimmers who stay out after an advisory
    lost_value_per_swim: float = 3.0  # dollars of enjoyment lost per deterred swim
    extra: dict = field(default_factory=dict)

    @property
    def benefit_true_alert(self) -> float:
        return (
            self.swimmers_per_site_day * self.compliance * self.illness_risk * self.cost_per_illness
        )

    @property
    def cost_false_alert(self) -> float:
        return self.swimmers_per_site_day * self.compliance * self.lost_value_per_swim

    @property
    def breakeven_probability(self) -> float:
        return self.cost_false_alert / (self.cost_false_alert + self.benefit_true_alert)
