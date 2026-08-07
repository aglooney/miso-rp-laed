"""Create a compact four-panel summary of the reviewer-motivated simulations.

The figure is intentionally based only on cases in which both RP-LMP and
LA-LMP serve load without shedding or spillage.  Run with:

    .venv/bin/python analysis/make_robustness_summary_figure.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "outputs" / "revision" / "aggregate_results.csv"
OUTPUT_PDF = ROOT / "images" / "robustness_summary.pdf"
OUTPUT_PNG = ROOT / "images" / "robustness_summary.png"
TOL = 1e-6

RP_COLOR = "#0072B2"
LA_COLOR = "#D55E00"
TLMP_COLOR = "#6B6B6B"


def _clean(frame: pd.DataFrame) -> pd.DataFrame:
    """Keep cases in which neither design uses emergency balancing."""
    return frame[
        (frame["emergency_rp_mwh"] <= TOL)
        & (frame["emergency_la_mwh"] <= TOL)
    ].copy()


def _thousands(value: float, _position: float | None = None) -> str:
    if abs(value) < 1e-9:
        return "0"
    return f"{value / 1_000:.0f}k"


def _ramp_panel(axis: plt.Axes, aggregate: pd.DataFrame) -> None:
    ramp_values = [0.1, 0.2, 0.4, 0.8]
    adder_values = [0.0, 0.08, 0.16, 0.24]
    ramp = aggregate[aggregate["study"] == "ramp_requirement"]
    shares = np.zeros((len(ramp_values), len(adder_values)))
    counts = np.zeros_like(shares, dtype=int)

    for row, alpha in enumerate(ramp_values):
        for column, adder in enumerate(adder_values):
            cases = ramp[
                np.isclose(ramp["ramp_multiplier"], alpha)
                & np.isclose(ramp["upward_adder_fraction"], adder)
            ]
            eligible = _clean(cases)
            counts[row, column] = len(eligible)
            shares[row, column] = (
                100
                * np.mean(eligible["loc_rp"] > eligible["loc_la"] + TOL)
                if len(eligible)
                else np.nan
            )

    image = axis.imshow(
        shares,
        cmap="Blues",
        norm=Normalize(vmin=0, vmax=100),
        aspect="auto",
    )
    axis.set_xticks(np.arange(len(adder_values)))
    axis.set_xticklabels([f"{100 * value:.0f}%" for value in adder_values])
    axis.set_yticks(np.arange(len(ramp_values)))
    axis.set_yticklabels([f"{value:.1f}" for value in ramp_values])
    axis.set_xlabel("Upward ramp-requirement adder")
    axis.set_ylabel("Ramp multiplier")
    axis.set_title("(a) Ramp stress and product requirement")

    for row in range(len(ramp_values)):
        for column in range(len(adder_values)):
            value = shares[row, column]
            foreground = "white" if value >= 55 else "#222222"
            axis.text(
                column,
                row,
                f"{value:.0f}%\n({counts[row, column]}/28)",
                ha="center",
                va="center",
                color=foreground,
                fontsize=7.2,
            )

    colorbar = axis.figure.colorbar(
        image, ax=axis, fraction=0.046, pad=0.03, ticks=[0, 50, 100]
    )
    colorbar.set_label("Eligible days with RP > LA")
    colorbar.ax.set_yticklabels(["0%", "50%", "100%"])


def _horizon_panel(axis: plt.Axes, aggregate: pd.DataFrame) -> None:
    cases = aggregate[aggregate["study"] == "external93_horizon"].copy()
    cases["clean"] = (
        (cases["emergency_rp_mwh"] <= TOL)
        & (cases["emergency_la_mwh"] <= TOL)
    )
    common_days = (
        cases.groupby(["month", "day"])["clean"]
        .all()
        .loc[lambda values: values]
        .index
    )
    case_index = pd.MultiIndex.from_frame(cases[["month", "day"]])
    cases = cases[case_index.isin(common_days)]
    medians = cases.groupby("horizon")[["loc_rp", "loc_la", "loc_tlmp"]].median()
    horizons = medians.index.to_numpy()

    axis.plot(
        horizons,
        medians["loc_rp"],
        "o-",
        color=RP_COLOR,
        label="RP-LMP",
        linewidth=1.8,
    )
    axis.plot(
        horizons,
        medians["loc_la"],
        "s-",
        color=LA_COLOR,
        label="LA-LMP",
        linewidth=1.8,
    )
    axis.plot(
        horizons,
        medians["loc_tlmp"],
        "^-",
        color=TLMP_COLOR,
        label="TLMP",
        linewidth=1.5,
    )
    for horizon, value in medians["loc_la"].items():
        axis.annotate(
            _thousands(value),
            (horizon, value),
            xytext=(0, -13 if horizon == 1 else 6),
            textcoords="offset points",
            ha="center",
            fontsize=7.2,
            color=LA_COLOR,
        )
    axis.set_xticks([1, 3, 7, 13])
    axis.set_ylim(-1_500, 58_000)
    axis.yaxis.set_major_formatter(FuncFormatter(_thousands))
    axis.set_xlabel("Look-ahead intervals (5 minutes each)")
    axis.set_ylabel("Median daily LOC ($)")
    axis.set_title("(b) Horizon ablation, 10 common RTS-93 days")
    axis.grid(axis="y", alpha=0.22)
    axis.legend(frameon=False, fontsize=7.5, loc="upper right")


def _forecast_panel(axis: plt.Axes, aggregate: pd.DataFrame) -> None:
    errors = [0.0, 0.01, 0.03, 0.05]
    forecast = aggregate[aggregate["study"] == "forecast_error"]
    rp_medians: list[float] = []
    la_medians: list[float] = []
    shares: list[float] = []
    eligible_counts: list[int] = []
    total_counts: list[int] = []

    for error in errors:
        cases = forecast[np.isclose(forecast["sigma_rel"], error)]
        eligible = _clean(cases)
        rp_medians.append(float(eligible["loc_rp"].median()))
        la_medians.append(float(eligible["loc_la"].median()))
        shares.append(
            100 * float(np.mean(eligible["loc_rp"] > eligible["loc_la"] + TOL))
        )
        eligible_counts.append(len(eligible))
        total_counts.append(len(cases))

    x = np.arange(len(errors))
    width = 0.36
    axis.bar(
        x - width / 2,
        rp_medians,
        width,
        color=RP_COLOR,
        label="RP-LMP",
    )
    axis.bar(
        x + width / 2,
        la_medians,
        width,
        color=LA_COLOR,
        label="LA-LMP",
    )
    axis.set_yscale("log")
    axis.set_ylim(25, 100_000)
    axis.set_xticks(x)
    axis.set_xticklabels([f"{100 * error:.0f}%" for error in errors])
    axis.set_xlabel("Base forecast-error scale")
    axis.set_ylabel("Median daily LOC ($, log scale)")
    axis.set_title(r"(c) Coherent AR(1) forecast error, $\rho=0.9$")
    axis.grid(axis="y", which="both", alpha=0.22)
    axis.legend(frameon=False, fontsize=7.5, loc="upper left")

    for index, share in enumerate(shares):
        top = max(rp_medians[index], la_medians[index])
        axis.annotate(
            f"RP>LA {share:.0f}%\n({eligible_counts[index]}/{total_counts[index]})",
            (index, top),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=6.8,
        )


def _external_panel(axis: plt.Axes, aggregate: pd.DataFrame) -> None:
    external = _clean(aggregate[aggregate["study"] == "external93"])
    ramp_values = [0.3, 0.6]
    x = np.arange(len(ramp_values))

    ordered_days = sorted(
        external[["month", "day"]].drop_duplicates().itertuples(index=False, name=None)
    )
    for month, day in ordered_days:
        day_rows = external[
            (external["month"] == month) & (external["day"] == day)
        ].set_index("ramp_multiplier")
        values = [
            float(day_rows.loc[alpha, "loc_rp"] - day_rows.loc[alpha, "loc_la"])
            for alpha in ramp_values
        ]
        axis.plot(x, values, color="#B3B3B3", linewidth=0.7, alpha=0.65, zorder=1)
        axis.scatter(x, values, color=RP_COLOR, s=15, alpha=0.75, zorder=2)

    medians = []
    for alpha in ramp_values:
        values = external[np.isclose(external["ramp_multiplier"], alpha)]
        relief = values["loc_rp"] - values["loc_la"]
        median = float(relief.median())
        medians.append(median)
        axis.plot(
            [x[ramp_values.index(alpha)] - 0.16, x[ramp_values.index(alpha)] + 0.16],
            [median, median],
            color=LA_COLOR,
            linewidth=2.4,
            zorder=3,
        )
        axis.annotate(
            f"median {_thousands(median)}",
            (x[ramp_values.index(alpha)], median),
            xytext=(0, 6),
            textcoords="offset points",
            ha="center",
            fontsize=7.2,
            color=LA_COLOR,
        )

    axis.axhline(0, color=TLMP_COLOR, linewidth=0.8)
    axis.set_xticks(x)
    axis.set_xticklabels([f"{value:.1f}" for value in ramp_values])
    axis.set_xlabel("Ramp multiplier")
    axis.set_ylabel("RP-LMP minus LA-LMP LOC ($)")
    axis.yaxis.set_major_formatter(FuncFormatter(_thousands))
    axis.set_title("(d) Full-day RTS-93 check, 12 days each")
    axis.grid(axis="y", alpha=0.22)
    axis.text(
        0.5,
        0.03,
        "All 24 differences are positive",
        transform=axis.transAxes,
        ha="center",
        va="bottom",
        fontsize=7.2,
    )


def main() -> None:
    aggregate = pd.read_csv(INPUT)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.titlesize": 8.5,
            "axes.labelsize": 8,
            "xtick.labelsize": 7.3,
            "ytick.labelsize": 7.3,
            "legend.fontsize": 7.3,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    figure, axes = plt.subplots(2, 2, figsize=(7.15, 4.65))
    _ramp_panel(axes[0, 0], aggregate)
    _horizon_panel(axes[0, 1], aggregate)
    _forecast_panel(axes[1, 0], aggregate)
    _external_panel(axes[1, 1], aggregate)
    figure.tight_layout(h_pad=1.5, w_pad=1.3)
    OUTPUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT_PDF, bbox_inches="tight")
    figure.savefig(OUTPUT_PNG, dpi=300, bbox_inches="tight")
    plt.close(figure)
    print(f"Wrote {OUTPUT_PDF}")
    print(f"Wrote {OUTPUT_PNG}")


if __name__ == "__main__":
    main()
