"""Create the 93-generator RTS-GMLC incidence figure used in main.tex."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def _r_squared(x: np.ndarray, y: np.ndarray) -> float:
    if np.std(x) <= 1e-12 or np.std(y) <= 1e-12:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1] ** 2)


def _fit_line(axis, x: np.ndarray, y: np.ndarray, color: str) -> None:
    if np.std(x) <= 1e-12:
        return
    grid = np.linspace(float(np.min(x)), float(np.max(x)), 100)
    coefficients = np.polyfit(x, y, 1)
    axis.plot(grid, np.polyval(coefficients, grid), color=color, linewidth=1.2)


def main(*, legend_style: str = "left-middle", output_suffix: str = "") -> None:
    panel = pd.read_csv(ROOT / "outputs" / "revision" / "generator_panel.csv")
    panel = panel[
        (panel["system"] == "RTS93")
        & (panel["study"] == "external93")
        & (panel["month"] == 10)
        & (panel["day"] == 15)
        & np.isclose(panel["ramp_multiplier"], 0.3)
        & np.isclose(panel["upward_adder_fraction"], 0.16)
    ].copy()
    if len(panel) != 93:
        raise AssertionError(f"Expected 93 generator rows, found {len(panel)}.")

    la_minus_rp = (panel["loc_la"] - panel["loc_rp"]).to_numpy()
    specifications = (
        (
            panel["rp_bind"].to_numpy(),
            "Bind frequency",
            "Bind frequency",
        ),
        (
            panel["rp_scarcity_flex"].to_numpy(),
            r"$f_g/(R_g/P_g^{\max})$",
            "Flex-adjusted bind exposure",
        ),
    )

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 8.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    figure, axes = plt.subplots(1, 2, figsize=(7.15, 2.55))
    rp_loc = panel["loc_rp"].to_numpy()
    la_loc = panel["loc_la"].to_numpy()
    legend_handles = None
    legend_labels = None
    left_delta_axis = None
    for axis, (x, xlabel, title) in zip(axes, specifications):
        axis.scatter(x, rp_loc, s=15, color="#0072B2", alpha=0.72, label="RP LOC")
        axis.scatter(x, la_loc, s=15, color="#D55E00", alpha=0.72, label="LA LOC")
        _fit_line(axis, x, rp_loc, "#0072B2")
        _fit_line(axis, x, la_loc, "#D55E00")
        axis.set_xlabel(xlabel)
        axis.set_ylabel(r"LOC (\$)")
        axis.grid(alpha=0.22, linewidth=0.5)

        delta_axis = axis.twinx()
        if left_delta_axis is None:
            left_delta_axis = delta_axis
        delta_axis.scatter(
            x,
            la_minus_rp,
            marker="^",
            facecolors="none",
            edgecolors="0.35",
            s=22,
            alpha=0.75,
            label=r"LA $-$ RP",
        )
        _fit_line(delta_axis, x, la_minus_rp, "0.4")
        delta_axis.lines[-1].set_linestyle("--")
        delta_axis.set_ylabel(r"$\mathrm{LOC}^{LA}-\mathrm{LOC}^{RP}$ (\$)")

        r2_rp = _r_squared(x, rp_loc)
        r2_la = _r_squared(x, la_loc)
        r2_difference = _r_squared(x, la_minus_rp)
        axis.set_title(
            f"{title}\n"
            + rf"$R^2$: RP {r2_rp:.3f}, LA {r2_la:.3f}, "
            + rf"LA$-$RP {r2_difference:.3f}"
        )

        handles, labels = axis.get_legend_handles_labels()
        delta_handles, delta_labels = delta_axis.get_legend_handles_labels()
        if legend_handles is None:
            legend_handles = handles + delta_handles
            legend_labels = labels + delta_labels

    if legend_style == "shared":
        figure.legend(
            legend_handles,
            legend_labels,
            frameon=False,
            fontsize=7.5,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.01),
            ncol=3,
            columnspacing=1.4,
            handletextpad=0.5,
        )
        figure.tight_layout(rect=(0.0, 0.14, 1.0, 1.0), w_pad=1.4)
    else:
        left_delta_axis.legend(
            legend_handles,
            legend_labels,
            frameon=True,
            framealpha=0.96,
            facecolor="white",
            edgecolor="0.75",
            fontsize=6.3,
            loc="center left",
            bbox_to_anchor=(0.015, 0.5),
            borderaxespad=0.0,
            borderpad=0.35,
            labelspacing=0.3,
            handletextpad=0.4,
            markerscale=0.85,
        )
        figure.tight_layout(w_pad=1.4)
    output_stem = ROOT / "images" / (
        "rts_dispatchable_laed_vs_rp_loc_bivar_ylinear" + output_suffix
    )
    figure.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(
        output_stem.with_suffix(".png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(figure)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--legend-style",
        choices=("shared", "left-middle"),
        default="left-middle",
    )
    parser.add_argument("--output-suffix", default="")
    arguments = parser.parse_args()
    main(
        legend_style=arguments.legend_style,
        output_suffix=arguments.output_suffix,
    )
