"""Plot coherent AR(1) forecasts and outcomes for reproducibly sampled days."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from data_inputs import load_rts_profiles, scale_profile, seasonal_days
from market_models import make_forecast_matrix


ROOT = Path(__file__).resolve().parents[1]
RTS_ROOT = ROOT / "data" / "RTS-GMLC"
AGGREGATE = ROOT / "outputs" / "revision" / "aggregate_results.csv"
OUTPUT_DIR = ROOT / "images"
SELECTION_PATH = ROOT / "outputs" / "revision" / "ar1_random_days.json"

SELECTION_SEED = 299
FORECAST_SEED = 11
SIGMA = 0.05
RHO = 0.9
HORIZON = 13
TOL = 1e-6

ACTUAL_COLOR = "#222222"
FORECAST_COLOR = "#D55E00"
CLEAN_COLOR = "#0072B2"
EMERGENCY_COLOR = "#B33A3A"
LINE_COLOR = "#8C8C8C"


def select_days() -> list[tuple[int, int]]:
    """Select two of the seven study days within each season."""
    rng = np.random.default_rng(SELECTION_SEED)
    by_month: dict[int, list[int]] = {}
    for month, day in seasonal_days():
        by_month.setdefault(month, []).append(day)
    selected: list[tuple[int, int]] = []
    for month in sorted(by_month):
        days = np.asarray(sorted(by_month[month]))
        chosen = sorted(rng.choice(days, size=2, replace=False).tolist())
        selected.extend((month, int(day)) for day in chosen)
    return selected


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.titlesize": 8.5,
            "axes.labelsize": 8,
            "xtick.labelsize": 7.2,
            "ytick.labelsize": 7.2,
            "legend.fontsize": 7.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def make_forecast_plot(selected: list[tuple[int, int]]) -> None:
    profiles = load_rts_profiles(RTS_ROOT)
    figure, axes = plt.subplots(2, 4, figsize=(7.15, 3.7), sharex=True)
    hours = np.arange(288) / 12.0

    for axis, (month, day) in zip(axes.flat, selected):
        actual_raw, day_ahead_raw = profiles[(month, day)]
        actual, _ = scale_profile(
            actual_raw, day_ahead_raw, target_mean=1_100.0
        )
        forecast = make_forecast_matrix(
            actual,
            max_horizon=HORIZON,
            sigma_rel=SIGMA,
            seed=FORECAST_SEED,
            error_model="ar1_delivery",
            ar1_rho=RHO,
        )
        axis.plot(hours, actual, color=ACTUAL_COLOR, linewidth=1.3, zorder=3)
        # Plot every one-hour rolling origin. Each segment covers the following
        # hour, including the exactly observed current interval.
        for origin in range(0, len(actual), 12):
            length = min(HORIZON, len(actual) - origin)
            segment_hours = (origin + np.arange(length)) / 12.0
            axis.plot(
                segment_hours,
                forecast[origin, :length],
                color=FORECAST_COLOR,
                linewidth=0.8,
                alpha=0.7,
                zorder=2,
            )
        axis.set_title(f"{month:02d}-{day:02d}")
        axis.grid(alpha=0.18)
        axis.set_xlim(0, 24)
        axis.set_xticks([0, 6, 12, 18, 24])

    for axis in axes[:, 0]:
        axis.set_ylabel("Net load (MW)")
    for axis in axes[-1, :]:
        axis.set_xlabel("Hour")
    figure.legend(
        handles=[
            Line2D(
                [0],
                [0],
                color=ACTUAL_COLOR,
                linewidth=1.4,
                label="Actual net load",
            ),
            Line2D(
                [0],
                [0],
                color=FORECAST_COLOR,
                linewidth=1.0,
                label="Hourly rolling net-load forecasts",
            ),
        ],
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 1.01),
    )
    figure.suptitle(
        r"Random seasonal days: coherent AR(1) forecasts "
        r"($\sigma=5\%$, $\rho=0.9$, seed 11)",
        y=1.06,
        fontsize=9,
    )
    figure.tight_layout(h_pad=1.0, w_pad=0.9)
    figure.savefig(
        OUTPUT_DIR / "ar1_random_days_forecasts.pdf", bbox_inches="tight"
    )
    figure.savefig(
        OUTPUT_DIR / "ar1_random_days_forecasts.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(figure)


def make_outcome_plot(selected: list[tuple[int, int]]) -> None:
    aggregate = pd.read_csv(AGGREGATE)
    forecast = aggregate[aggregate["study"] == "forecast_error"].copy()
    forecast["clean"] = (
        (forecast["emergency_rp_mwh"] <= TOL)
        & (forecast["emergency_la_mwh"] <= TOL)
    )
    forecast["relief"] = forecast["loc_rp"] - forecast["loc_la"]

    figure, axes = plt.subplots(
        2, 4, figsize=(7.15, 3.8), sharex=True, sharey=True
    )
    error_levels = [0.0, 0.01, 0.03, 0.05]
    x = 100 * np.asarray(error_levels)

    for axis, (month, day) in zip(axes.flat, selected):
        day_rows = forecast[
            (forecast["month"] == month) & (forecast["day"] == day)
        ]
        perfect = day_rows[np.isclose(day_rows["sigma_rel"], 0.0)].iloc[0]
        for seed in (11, 22, 33):
            values = [float(perfect["relief"])]
            clean = [bool(perfect["clean"])]
            for sigma in error_levels[1:]:
                row = day_rows[
                    np.isclose(day_rows["sigma_rel"], sigma)
                    & (day_rows["seed"] == seed)
                ].iloc[0]
                values.append(float(row["relief"]))
                clean.append(bool(row["clean"]))
            axis.plot(x, values, color=LINE_COLOR, linewidth=0.75, alpha=0.7)
            for xpos, value, is_clean in zip(x, values, clean):
                axis.scatter(
                    xpos,
                    value,
                    s=18,
                    marker="o",
                    facecolor=CLEAN_COLOR if is_clean else "none",
                    edgecolor=CLEAN_COLOR if is_clean else EMERGENCY_COLOR,
                    linewidth=1.0,
                    zorder=3,
                )
        clean_count = int(day_rows["clean"].sum())
        axis.set_title(f"{month:02d}-{day:02d}  |  clean {clean_count}/10")
        axis.axhline(0, color=ACTUAL_COLOR, linewidth=0.7)
        axis.grid(axis="y", alpha=0.18)
        axis.set_yscale("symlog", linthresh=100)
        axis.set_xticks(x)
        axis.set_xticklabels(["0", "1", "3", "5"])

    for axis in axes[:, 0]:
        axis.set_ylabel(
            r"$\mathrm{LOC}^{\mathrm{RP}}-\mathrm{LOC}^{\mathrm{LA}}$ (\$)"
        )
    for axis in axes[-1, :]:
        axis.set_xlabel("Base forecast error (%)")
    axes[0, 0].text(
        0.02,
        0.95,
        "LA lower",
        transform=axes[0, 0].transAxes,
        ha="left",
        va="top",
        fontsize=7,
    )
    axes[0, 0].text(
        0.02,
        0.05,
        "RP lower",
        transform=axes[0, 0].transAxes,
        ha="left",
        va="bottom",
        fontsize=7,
    )
    figure.legend(
        handles=[
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor=CLEAN_COLOR,
                markeredgecolor=CLEAN_COLOR,
                label="Emergency-free",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor="none",
                markeredgecolor=EMERGENCY_COLOR,
                label="Shedding/spillage in either design",
            ),
        ],
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 1.01),
    )
    figure.suptitle(
        r"AR(1) forecast-error outcomes on the same random days "
        r"($\rho=0.9$; three seeds at nonzero error)",
        y=1.06,
        fontsize=9,
    )
    figure.tight_layout(h_pad=1.0, w_pad=0.9)
    figure.savefig(
        OUTPUT_DIR / "ar1_random_days_outcomes.pdf", bbox_inches="tight"
    )
    figure.savefig(
        OUTPUT_DIR / "ar1_random_days_outcomes.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(figure)


def main() -> None:
    configure_style()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    selected = select_days()
    SELECTION_PATH.write_text(
        json.dumps(
            {
                "selection_seed": SELECTION_SEED,
                "selection_rule": "two of seven days sampled within each month",
                "days": [f"{month:02d}-{day:02d}" for month, day in selected],
                "forecast_seed": FORECAST_SEED,
                "sigma_rel": SIGMA,
                "ar1_rho": RHO,
            },
            indent=2,
        )
        + "\n"
    )
    make_forecast_plot(selected)
    make_outcome_plot(selected)
    print("Selected:", ", ".join(f"{m:02d}-{d:02d}" for m, d in selected))
    print(f"Wrote {OUTPUT_DIR / 'ar1_random_days_forecasts.pdf'}")
    print(f"Wrote {OUTPUT_DIR / 'ar1_random_days_outcomes.pdf'}")


if __name__ == "__main__":
    main()
