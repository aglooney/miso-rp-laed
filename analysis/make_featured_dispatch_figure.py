"""Recreate the featured 10-generator RP/LA dispatch comparison."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from data_inputs import ten_generator_fleet
from market_models import make_forecast_matrix, simulate_day


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    with (ROOT / "MISO_Projection.json").open() as stream:
        raw = json.load(stream)["2032_Aug"]
    load = np.asarray([raw[key] for key in sorted(raw, key=int)], dtype=float)
    load *= 1100.0 / load.mean()

    # A 13-interval rolling window implements the first 276 intervals for the
    # full-horizon comparison used in the featured analysis.
    load = load[: len(load) - 12]
    fleet = ten_generator_fleet().with_ramp_multiplier(0.1)
    forecasts = make_forecast_matrix(load, max_horizon=13)
    simulation = simulate_day(
        fleet=fleet,
        actual_load=load,
        forecast_matrix=forecasts,
        horizon=13,
        upward_adder_fraction=40.0 / 1100.0,
        shortage_penalty=65.0,
    )

    hours = np.arange(len(load)) / 12.0
    generators = (10, 6, 2, 7)
    colors = ("#0072B2", "#D55E00", "#009E73", "#CC79A7")

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 8.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    # Build at the final single-column width so the lettering is not reduced
    # when LaTeX places the figure at \columnwidth.
    figure, axes = plt.subplots(2, 1, figsize=(3.45, 3.70), sharex=True)
    net_load_handle = None
    for axis, field, label in zip(
        axes, ("rp_p", "la_p"), ("(a) RP-LMP", "(b) LA-LMP")
    ):
        for generator, color in zip(generators, colors):
            axis.plot(
                hours,
                simulation[field][generator - 1],
                color=color,
                linewidth=1.15,
                label=f"G{generator}",
            )
        axis.set_ylim(-10, 750)
        axis.set_ylabel("Generation (MW)")
        axis.text(
            0.01,
            0.80,
            label,
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontweight="bold",
        )
        axis.grid(alpha=0.2, linewidth=0.4)

        load_axis = axis.twinx()
        load_line, = load_axis.plot(
            hours,
            load,
            color="0.25",
            linestyle="--",
            linewidth=1.0,
            label="Net load",
        )
        if net_load_handle is None:
            net_load_handle = load_line
        load_axis.set_ylim(600, 2100)
        load_axis.set_ylabel("Load (MW)")

    generator_handles, generator_labels = axes[0].get_legend_handles_labels()
    figure.legend(
        generator_handles + [net_load_handle],
        generator_labels + ["Net load"],
        ncol=3,
        frameon=False,
        fontsize=8.5,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        columnspacing=1.0,
        handlelength=1.8,
        handletextpad=0.45,
    )
    axes[-1].set_xlabel("Hour")
    axes[-1].set_xlim(0, hours[-1])
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.86), h_pad=0.35)
    figure.savefig(
        ROOT / "images" / "10gen_dispatch_window_rp_vs_la.pdf",
        bbox_inches="tight",
    )
    plt.close(figure)


if __name__ == "__main__":
    main()
