"""Nearest-site lookup with shapely, used for the 'what did the neighbours read' features."""

from __future__ import annotations

import numpy as np
import pandas as pd
from shapely import STRtree
from shapely.geometry import Point

KM_PER_DEG_LAT = 110.57


def project_km(
    lat: np.ndarray, lon: np.ndarray, lat0: float = -33.85
) -> tuple[np.ndarray, np.ndarray]:
    """Local equirectangular projection. Good to a fraction of a percent over Greater Sydney."""
    x = np.asarray(lon) * 111.32 * np.cos(np.radians(lat0))
    y = np.asarray(lat) * KM_PER_DEG_LAT
    return x, y


def site_table(df: pd.DataFrame) -> pd.DataFrame:
    """One row per site with a stable position (median of the reported coordinates)."""
    tab = df.groupby("swim_site").agg(
        lat=("latitude", "median"), lon=("longitude", "median"), region=("region", "first")
    )
    x, y = project_km(tab["lat"].to_numpy(), tab["lon"].to_numpy())
    tab["x_km"], tab["y_km"] = x, y
    return tab


def nearest_sites(tab: pd.DataFrame, k: int = 3) -> dict[str, list[tuple[str, float]]]:
    """Return the k nearest other sites for every site as (name, distance_km) pairs."""
    names = tab.index.to_numpy()
    points = [Point(x, y) for x, y in zip(tab["x_km"], tab["y_km"], strict=True)]
    tree = STRtree(points)
    out: dict[str, list[tuple[str, float]]] = {}
    for i, pt in enumerate(points):
        # A 25 km window covers the whole harbour and coast, so k neighbours always exist.
        cand_idx = tree.query(pt.buffer(25.0))
        ranked = sorted(
            ((int(j), pt.distance(points[j])) for j in cand_idx if j != i), key=lambda t: t[1]
        )[:k]
        out[str(names[i])] = [(str(names[j]), float(d)) for j, d in ranked]
    return out
