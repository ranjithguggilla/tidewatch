"""Gradio front end. All the logic lives in score.score_site_day; this file only wires inputs."""

from __future__ import annotations

from pathlib import Path

import gradio as gr
import joblib
import pandas as pd

from .config import ROOT
from .score import score_site_day


def parse_rain(text: str) -> list[float]:
    """Parse '12, 0, 3.5' into floats, most recent day first. Blank entries are ignored."""
    vals = []
    for part in text.replace(";", ",").split(","):
        part = part.strip()
        if part:
            vals.append(float(part))
    return vals


def run_score(
    bundle: dict,
    site: str,
    date: str,
    rain_text: str,
    tmax: float,
    tmin: float,
    last_result: float | None,
):
    daily = parse_rain(rain_text)
    out = score_site_day(
        bundle,
        site,
        pd.Timestamp(date),
        daily,
        tmax,
        tmin,
        None if last_result in (None, "") else float(last_result),
    )
    reasons = (
        "\n".join(f"- {r.text}" for r in out["reasons"])
        or "- No single factor moves the score much."
    )
    summary = (
        f"**{out['level']}**\n\n"
        f"Chance this sample exceeds {bundle['exceed_cfu']:.0f} per 100 mL: "
        f"**{out['probability'] * 100:.1f}%** "
        f"(this site's long-run rate is {out['site_baseline'] * 100:.1f}%).\n\n"
        f"What is driving it:\n{reasons}"
    )
    return summary


def build_app(bundle: dict) -> gr.Blocks:
    sites = sorted(bundle["site_state"].index.tolist())
    with gr.Blocks(title="tidewatch") as demo:
        gr.Markdown(
            "# tidewatch\nEnter the rain record for the days before a sample and see how likely "
            "the water is to fail. Rain is listed most recent day first."
        )
        with gr.Row():
            site = gr.Dropdown(sites, value=sites[0], label="Swim site")
            date = gr.Textbox(value="2025-01-10", label="Sample date (YYYY-MM-DD)")
        rain = gr.Textbox(value="22, 31, 4, 0, 0, 0, 0", label="Daily rain in mm, yesterday first")
        with gr.Row():
            tmax = gr.Number(value=26.0, label="Yesterday's max temperature (C)")
            tmin = gr.Number(value=19.0, label="Yesterday's min temperature (C)")
            last = gr.Number(
                value=None, label="Last lab result at this site (optional, per 100 mL)"
            )
        btn = gr.Button("Score")
        out = gr.Markdown()
        btn.click(lambda *a: run_score(bundle, *a), [site, date, rain, tmax, tmin, last], out)
    return demo


def main(model_path: Path | None = None) -> None:
    path = model_path or ROOT / "outputs" / "model.joblib"
    bundle = joblib.load(path)
    build_app(bundle).launch()


if __name__ == "__main__":
    main()
