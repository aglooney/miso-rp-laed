"""Compare forecast-error exposure with matched ramp and LAED horizons.

This is a standalone diagnostic; it does not modify the manuscript tables.
For every day/error/seed combination, all designs receive the same AR(1)
delivery-time forecast matrix:

* RP10: one cumulative ramp product using the 10-minute forecast.
* RP10+30: nested cumulative products using the 10- and 30-minute forecasts.
* LA10, LA30, LA60: look-ahead dispatch over 3, 7, and 13 five-minute
  intervals, respectively (the current interval plus 10, 30, or 60 minutes).

The output tables report interquartile ranges on a common sample for which
all five designs have zero shedding and spillage.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from data_inputs import (
    external_check_days,
    load_rts_profiles,
    rts_controllable_fleet,
    scale_profile,
    seasonal_days,
    ten_generator_fleet,
)
from market_models import make_forecast_matrix, simulate_day


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RTS = ROOT / "data" / "RTS-GMLC"
OUTPUT_DIR = ROOT / "outputs" / "forecast_horizon"
REVISION_AGGREGATE_PATH = ROOT / "outputs" / "revision" / "aggregate_results.csv"
CASE_PATH = OUTPUT_DIR / "forecast_horizon_cases.csv"
SUMMARY_PATH = OUTPUT_DIR / "forecast_horizon_iqr.csv"
MARKDOWN_PATH = OUTPUT_DIR / "forecast_horizon_iqr.md"
LATEX_PATH = OUTPUT_DIR / "forecast_horizon_iqr.tex"
METADATA_PATH = OUTPUT_DIR / "forecast_horizon_metadata.json"

DESIGNS = ("rp10", "rp10_30", "la10", "la30", "la60")
DESIGN_LABELS = {
    "rp10": "RP 10 min",
    "rp10_30": "RP 10+30 min",
    "la10": "LA 10 min",
    "la30": "LA 30 min",
    "la60": "LA 60 min",
}
ERROR_LEVELS = (0.0, 0.01, 0.03, 0.05)
NONZERO_SEEDS = (11, 22, 33)


def _case_key(row: dict[str, Any]) -> tuple[str, int, int, float, int]:
    return (
        str(row["system"]),
        int(row["month"]),
        int(row["day"]),
        float(row["sigma_rel"]),
        int(row["seed"]),
    )


def _emergency_mwh(sim: dict[str, Any], design: str) -> float:
    dt = float(sim["dt_hours"])
    if design.startswith("rp"):
        return float(np.sum(sim["rp_shed"] + sim["rp_spill"]) * dt)
    return float(np.sum(sim["la_shed"] + sim["la_spill"]) * dt)


def _loc(sim: dict[str, Any], design: str) -> float:
    if design.startswith("rp"):
        return float(np.sum(sim["rp_loc"]))
    return float(np.sum(sim["la_loc"]))


def _run_path(
    *,
    fleet,
    actual: np.ndarray,
    forecast: np.ndarray,
    ramp_multiplier: float,
    uncertainty_proxy_fraction: float,
    shortage_penalty: float,
    reference: dict[str, float],
) -> dict[str, float]:
    scaled_fleet = fleet.with_ramp_multiplier(ramp_multiplier)

    # This run supplies RP10 and LA10.
    sim_10 = simulate_day(
        fleet=scaled_fleet,
        actual_load=actual,
        forecast_matrix=forecast,
        horizon=3,
        upward_adder_fraction=uncertainty_proxy_fraction,
        product_intervals=2,
        shortage_penalty=shortage_penalty,
    )
    # This run supplies the nested RP10+30 design and LA30.
    sim_30 = simulate_day(
        fleet=scaled_fleet,
        actual_load=actual,
        forecast_matrix=forecast,
        horizon=7,
        upward_adder_fraction=(
            uncertainty_proxy_fraction,
            uncertainty_proxy_fraction,
        ),
        product_intervals=(2, 6),
        shortage_penalty=shortage_penalty,
    )
    rp10_loc = _loc(sim_10, "rp10")
    rp10_check = float(reference["loc_rp"])
    if not np.isclose(rp10_loc, rp10_check, rtol=1e-9, atol=1e-6):
        raise AssertionError(
            f"RP10 differs from the audited revision result: "
            f"{rp10_loc} versus {rp10_check}."
        )
    rp10_emergency = _emergency_mwh(sim_10, "rp10")
    rp10_emergency_check = float(reference["emergency_rp_mwh"])
    if not np.isclose(
        rp10_emergency, rp10_emergency_check, rtol=1e-9, atol=1e-9
    ):
        raise AssertionError("RP10 emergency energy differs from the audited result.")

    results = {
        "loc_rp10": rp10_loc,
        "loc_rp10_30": _loc(sim_30, "rp10_30"),
        "loc_la10": _loc(sim_10, "la10"),
        "loc_la30": _loc(sim_30, "la30"),
        "loc_la60": float(reference["loc_la"]),
        "emergency_rp10_mwh": rp10_emergency,
        "emergency_rp10_30_mwh": _emergency_mwh(sim_30, "rp10_30"),
        "emergency_la10_mwh": _emergency_mwh(sim_10, "la10"),
        "emergency_la30_mwh": _emergency_mwh(sim_30, "la30"),
        "emergency_la60_mwh": float(reference["emergency_la_mwh"]),
        "shortage_rp10_mwh": float(
            np.sum(
                sim_10["rp_short_up_products"]
                + sim_10["rp_short_down_products"]
            )
            * sim_10["dt_hours"]
        ),
        "shortage_rp10_30_mwh": float(
            np.sum(
                sim_30["rp_short_up_products"]
                + sim_30["rp_short_down_products"]
            )
            * sim_30["dt_hours"]
        ),
    }
    for design in DESIGNS:
        results[f"relief_rp10_vs_{design}"] = (
            results["loc_rp10"] - results[f"loc_{design}"]
        )
    results["clean_all"] = float(
        all(results[f"emergency_{design}_mwh"] <= 1e-6 for design in DESIGNS)
    )
    return results


def _iqr(series: pd.Series) -> tuple[float, float]:
    if series.empty:
        return float("nan"), float("nan")
    q1, q3 = np.quantile(series.to_numpy(dtype=float), [0.25, 0.75])
    return float(q1), float(q3)


def _summarize(cases: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (system, sigma), group in cases.groupby(
        ["system", "sigma_rel"], sort=True
    ):
        clean = group[group["clean_all"] > 0.5]
        row: dict[str, Any] = {
            "system": system,
            "error_percent": int(round(100.0 * sigma)),
            "clean_cases": int(len(clean)),
            "total_cases": int(len(group)),
        }
        for design in DESIGNS:
            q1, q3 = _iqr(clean[f"loc_{design}"])
            row[f"loc_{design}_q1"] = q1
            row[f"loc_{design}_q3"] = q3
            relief_q1, relief_q3 = _iqr(
                clean[f"relief_rp10_vs_{design}"]
            )
            row[f"relief_{design}_q1"] = relief_q1
            row[f"relief_{design}_q3"] = relief_q3
        rows.append(row)
    return pd.DataFrame(rows)


def _money(value: float) -> str:
    if not np.isfinite(value):
        return "--"
    rounded = int(np.rint(value))
    return f"{rounded:,}"


def _interval(row: pd.Series, prefix: str) -> str:
    return (
        f"[{_money(float(row[f'{prefix}_q1']))}, "
        f"{_money(float(row[f'{prefix}_q3']))}]"
    )


def _write_tables(summary: pd.DataFrame) -> None:
    markdown: list[str] = [
        "# Forecast-horizon comparison",
        "",
        "Every cell is an interquartile range in dollars. Calculations use the "
        "same emergency-free cases across all five designs within a row.",
        "The 10-generator panel uses ramp multiplier 0.2 and uncertainty proxy "
        "fraction 0.08; RTS-93 uses 0.6 and 0.16. Every design receives the "
        "same AR(1) delivery-time error path with rho 0.9.",
        "",
        "## Total LOC IQR",
        "",
        "| System | AR(1) error | No emergency | RP 10 min | RP 10+30 min | "
        "LA 10 min | LA 30 min | LA 60 min |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in summary.iterrows():
        markdown.append(
            f"| {row['system']} | {int(row['error_percent'])}% | "
            f"{int(row['clean_cases'])}/{int(row['total_cases'])} | "
            f"{_interval(row, 'loc_rp10')} | "
            f"{_interval(row, 'loc_rp10_30')} | "
            f"{_interval(row, 'loc_la10')} | "
            f"{_interval(row, 'loc_la30')} | "
            f"{_interval(row, 'loc_la60')} |"
        )
    markdown.extend(
        [
            "",
            "## Paired relief IQR relative to RP 10 min",
            "",
            "Positive relief means the column design has lower LOC than RP 10 min.",
            "",
            "| System | AR(1) error | RP 10+30 min | LA 10 min | LA 30 min | "
            "LA 60 min |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for _, row in summary.iterrows():
        markdown.append(
            f"| {row['system']} | {int(row['error_percent'])}% | "
            f"{_interval(row, 'relief_rp10_30')} | "
            f"{_interval(row, 'relief_la10')} | "
            f"{_interval(row, 'relief_la30')} | "
            f"{_interval(row, 'relief_la60')} |"
        )
    markdown.extend(
        [
            "",
            "RP 10+30 uses nested cumulative requirements. The first capability "
            "segment is deliverable within 10 minutes; the additional segment "
            "is deliverable between 10 and 30 minutes. Cumulative headroom and "
            "footroom prevent physical double-counting. The same stylized "
            "uncertainty proxy is added to each cumulative requirement. This "
            "diagnostic has not been inserted into the manuscript.",
            "",
        ]
    )
    MARKDOWN_PATH.write_text("\n".join(markdown))

    latex: list[str] = [
        r"\begin{table*}[!t]",
        r"\caption{Forecast-horizon comparison. Each entry is the total-LOC "
        r"IQR in dollars on the common no-emergency sample.}",
        r"\label{tab:forecast_horizon_diagnostic}",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{2.8pt}",
        r"\begin{tabular}{llrrrrrr}",
        r"\toprule",
        r"System & Error & No emerg. & RP 10 & RP 10+30 & LA 10 & LA 30 & LA 60 \\",
        r"\midrule",
    ]
    for _, row in summary.iterrows():
        fields = [
            str(row["system"]),
            f"{int(row['error_percent'])}\\%",
            f"{int(row['clean_cases'])}/{int(row['total_cases'])}",
            _interval(row, "loc_rp10"),
            _interval(row, "loc_rp10_30"),
            _interval(row, "loc_la10"),
            _interval(row, "loc_la30"),
            _interval(row, "loc_la60"),
        ]
        latex.append(" & ".join(fields) + r" \\")
    latex.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""])
    LATEX_PATH.write_text("\n".join(latex))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rts-root", type=Path, default=DEFAULT_RTS)
    parser.add_argument(
        "--system",
        choices=("10gen", "RTS93", "both"),
        default="both",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use one day and one nonzero seed per selected system.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Ignore checkpointed cases and rebuild selected systems.",
    )
    args = parser.parse_args()

    if not (args.rts_root / "RTS_Data").exists():
        raise SystemExit(f"RTS-GMLC data not found at {args.rts_root}.")
    if not REVISION_AGGREGATE_PATH.exists():
        raise SystemExit(
            "Audited revision aggregate results are required for the LA60 "
            f"reference: {REVISION_AGGREGATE_PATH}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    profiles = load_rts_profiles(args.rts_root)
    controllable_profiles = load_rts_profiles(
        args.rts_root, subtract_hydro=False
    )
    systems = ("10gen", "RTS93") if args.system == "both" else (args.system,)
    revision = pd.read_csv(REVISION_AGGREGATE_PATH)
    settings = {
        "10gen": {
            "days": seasonal_days(),
            "fleet": ten_generator_fleet(),
            "profiles": profiles,
            "ramp_multiplier": 0.2,
            "uncertainty_proxy_fraction": 0.08,
        },
        "RTS93": {
            "days": external_check_days(),
            "fleet": rts_controllable_fleet(args.rts_root),
            "profiles": controllable_profiles,
            "ramp_multiplier": 0.6,
            "uncertainty_proxy_fraction": 0.16,
        },
    }
    if args.quick:
        for system in systems:
            settings[system]["days"] = settings[system]["days"][:1]

    existing = pd.read_csv(CASE_PATH) if CASE_PATH.exists() else pd.DataFrame()
    if args.force and not existing.empty:
        existing = existing[~existing["system"].isin(systems)].copy()
    completed = {
        _case_key(row) for row in existing.to_dict("records")
    } if not existing.empty else set()
    rows = existing.to_dict("records") if not existing.empty else []

    total = 0
    for system in systems:
        day_count = len(settings[system]["days"])
        total += day_count * (1 + 3 * len(ERROR_LEVELS[1:]))
    finished = 0
    for system in systems:
        config = settings[system]
        for month, day in config["days"]:
            actual_raw, forecast_raw = config["profiles"][(month, day)]
            if system == "10gen":
                actual, _ = scale_profile(
                    actual_raw, forecast_raw, target_mean=1100.0
                )
            else:
                actual = actual_raw
            for sigma in ERROR_LEVELS:
                seeds = (0,) if sigma == 0.0 else NONZERO_SEEDS
                if args.quick and sigma > 0.0:
                    seeds = seeds[:1]
                for seed in seeds:
                    identity = (system, month, day, sigma, seed)
                    if identity in completed:
                        finished += 1
                        continue
                    forecast = make_forecast_matrix(
                        actual,
                        max_horizon=13,
                        sigma_rel=sigma,
                        seed=seed,
                        error_model="ar1_delivery",
                        ar1_rho=0.9,
                    )
                    study = (
                        "forecast_error"
                        if system == "10gen"
                        else "external93_forecast_error"
                    )
                    reference_rows = revision[
                        (revision["system"] == system)
                        & (revision["study"] == study)
                        & (revision["month"] == month)
                        & (revision["day"] == day)
                        & np.isclose(revision["sigma_rel"], sigma)
                        & (revision["seed"] == seed)
                    ]
                    if len(reference_rows) != 1:
                        raise AssertionError(
                            f"Expected one audited reference row for {identity}; "
                            f"found {len(reference_rows)}."
                        )
                    reference = reference_rows.iloc[0].to_dict()
                    result = _run_path(
                        fleet=config["fleet"],
                        actual=actual,
                        forecast=forecast,
                        ramp_multiplier=float(config["ramp_multiplier"]),
                        uncertainty_proxy_fraction=float(
                            config["uncertainty_proxy_fraction"]
                        ),
                        shortage_penalty=65.0,
                        reference=reference,
                    )
                    rows.append(
                        {
                            "system": system,
                            "month": month,
                            "day": day,
                            "sigma_rel": sigma,
                            "seed": seed,
                            "forecast_model": "ar1_delivery",
                            "forecast_rho": 0.9,
                            "ramp_multiplier": config["ramp_multiplier"],
                            "uncertainty_proxy_fraction": config[
                                "uncertainty_proxy_fraction"
                            ],
                            **result,
                        }
                    )
                    completed.add(identity)
                    finished += 1
                    pd.DataFrame(rows).to_csv(CASE_PATH, index=False)
                    print(
                        f"completed {finished}/{total}: {system} "
                        f"{month:02d}-{day:02d}, error={100*sigma:.0f}%, seed={seed}",
                        flush=True,
                    )

    cases = pd.DataFrame(rows)
    selected = cases[cases["system"].isin(systems)].copy()
    summary = _summarize(selected)
    summary.to_csv(SUMMARY_PATH, index=False)
    _write_tables(summary)
    metadata = {
        "interval_minutes": 5,
        "forecast_model": "ar1_delivery",
        "forecast_rho": 0.9,
        "common_random_numbers": True,
        "la60_reference": str(REVISION_AGGREGATE_PATH.relative_to(ROOT)),
        "designs": {
            "rp10": {"product_intervals": [2], "forecast_leads": [2]},
            "rp10_30": {
                "product_intervals": [2, 6],
                "forecast_leads": [2, 6],
                "nested_cumulative_awards": True,
            },
            "la10": {"lookahead_intervals": 3, "last_forecast_lead": 2},
            "la30": {"lookahead_intervals": 7, "last_forecast_lead": 6},
            "la60": {"lookahead_intervals": 13, "last_forecast_lead": 12},
        },
        "systems": {
            system: {
                "ramp_multiplier": settings[system]["ramp_multiplier"],
                "uncertainty_proxy_fraction": settings[system][
                    "uncertainty_proxy_fraction"
                ],
                "days": [
                    f"{month:02d}-{day:02d}"
                    for month, day in settings[system]["days"]
                ],
            }
            for system in systems
        },
        "common_sample_rule": (
            "zero shedding and spillage in all five designs, threshold 1e-6 MWh"
        ),
        "case_rows": int(len(selected)),
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"wrote {SUMMARY_PATH}")
    print(f"wrote {MARKDOWN_PATH}")


if __name__ == "__main__":
    main()
