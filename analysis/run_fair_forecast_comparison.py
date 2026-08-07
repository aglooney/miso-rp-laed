"""Run a mechanism-isolation forecast-error comparison.

Primary fairness choices
------------------------
* The same nested AR(1) revision path is supplied to every design.
* Reported error levels are 30--60 minute RMSE targets.  The one-step
  innovation scale is half that target, giving 10-minute RMSE equal to the
  target divided by sqrt(2).
* Ramp products have no uncertainty buffer.
* A 3,000 $/MWh numerical feasibility slack approximates a hard product
  requirement.  Any path using positive product slack is classified as
  infeasible and excluded from the common-sample IQRs.
* LAED is classified as forecast-feasible only when none of its rolling
  look-ahead solutions uses planned shedding or spillage.

This diagnostic is separate from the manuscript and the market-like soft
requirement sensitivity in ``run_forecast_horizon_comparison.py``.
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
OUTPUT_DIR = ROOT / "outputs" / "forecast_horizon_fair"
CASE_PATH = OUTPUT_DIR / "fair_forecast_cases.csv"
SUMMARY_PATH = OUTPUT_DIR / "fair_forecast_iqr.csv"
MARKDOWN_PATH = OUTPUT_DIR / "fair_forecast_iqr.md"
LATEX_PATH = OUTPUT_DIR / "fair_forecast_iqr.tex"
METADATA_PATH = OUTPUT_DIR / "fair_forecast_metadata.json"

DESIGNS = ("rp10", "rp10_30", "la10", "la30", "la60")
ERROR_TARGETS = (0.0, 0.01, 0.03, 0.05)
NONZERO_SEEDS = (11, 22, 33)
FEASIBILITY_PENALTY = 3000.0
TOL = 1e-6


def _key(row: dict[str, Any]) -> tuple[str, int, int, float, int]:
    return (
        str(row["system"]),
        int(row["month"]),
        int(row["day"]),
        float(row["long_rmse_target"]),
        int(row["seed"]),
    )


def _loc(sim: dict[str, Any], market: str) -> float:
    values = sim["rp_loc"] if market == "rp" else sim["la_loc"]
    return float(np.sum(values))


def _actual_emergency(sim: dict[str, Any], market: str) -> float:
    dt = float(sim["dt_hours"])
    if market == "rp":
        values = sim["rp_shed"] + sim["rp_spill"]
    else:
        values = sim["la_shed"] + sim["la_spill"]
    return float(np.sum(values) * dt)


def _planned_la_emergency(sim: dict[str, Any]) -> float:
    return float(
        max(
            np.max(sim["la_planned_shed_max"]),
            np.max(sim["la_planned_spill_max"]),
        )
    )


def _product_shortage(sim: dict[str, Any]) -> float:
    return float(
        np.sum(
            sim["rp_short_up_products"] + sim["rp_short_down_products"]
        )
        * sim["dt_hours"]
    )


def _relative_rmse(
    actual: np.ndarray, forecast: np.ndarray, lead: int
) -> float:
    origins = np.arange(len(actual))
    delivery = np.minimum(origins + lead, len(actual) - 1)
    denominator = np.maximum(actual[delivery], 1.0)
    relative_error = (forecast[:, lead] - actual[delivery]) / denominator
    return float(np.sqrt(np.mean(relative_error**2)))


def _run_path(
    *,
    fleet,
    actual: np.ndarray,
    forecast: np.ndarray,
    ramp_multiplier: float,
) -> dict[str, float]:
    scaled_fleet = fleet.with_ramp_multiplier(ramp_multiplier)
    common = dict(
        fleet=scaled_fleet,
        actual_load=actual,
        forecast_matrix=forecast,
        shortage_penalty=FEASIBILITY_PENALTY,
    )
    sim10 = simulate_day(
        **common,
        horizon=3,
        product_intervals=2,
        upward_adder_fraction=0.0,
    )
    sim30 = simulate_day(
        **common,
        horizon=7,
        product_intervals=(2, 6),
        upward_adder_fraction=(0.0, 0.0),
    )
    sim60 = simulate_day(
        **common,
        horizon=13,
        product_intervals=2,
        upward_adder_fraction=0.0,
    )

    if not np.isclose(
        _loc(sim10, "rp"), _loc(sim60, "rp"), rtol=1e-9, atol=1e-6
    ):
        raise AssertionError("RP10 changed with the independently solved LA horizon.")

    result = {
        "loc_rp10": _loc(sim10, "rp"),
        "loc_rp10_30": _loc(sim30, "rp"),
        "loc_la10": _loc(sim10, "la"),
        "loc_la30": _loc(sim30, "la"),
        "loc_la60": _loc(sim60, "la"),
        "emergency_rp10_mwh": _actual_emergency(sim10, "rp"),
        "emergency_rp10_30_mwh": _actual_emergency(sim30, "rp"),
        "emergency_la10_mwh": _actual_emergency(sim10, "la"),
        "emergency_la30_mwh": _actual_emergency(sim30, "la"),
        "emergency_la60_mwh": _actual_emergency(sim60, "la"),
        "planned_emergency_la10_mw": _planned_la_emergency(sim10),
        "planned_emergency_la30_mw": _planned_la_emergency(sim30),
        "planned_emergency_la60_mw": _planned_la_emergency(sim60),
        "shortage_rp10_mwh": _product_shortage(sim10),
        "shortage_rp10_30_mwh": _product_shortage(sim30),
        "max_product_price_rp10": float(
            np.max(
                np.r_[
                    sim10["rp_nu_up_products"].ravel(),
                    sim10["rp_nu_down_products"].ravel(),
                ]
            )
        ),
        "max_product_price_rp10_30": float(
            np.max(
                np.r_[
                    sim30["rp_nu_up_products"].ravel(),
                    sim30["rp_nu_down_products"].ravel(),
                ]
            )
        ),
    }
    result["feasible_rp10"] = float(
        result["shortage_rp10_mwh"] <= TOL
        and result["emergency_rp10_mwh"] <= TOL
    )
    result["feasible_rp10_30"] = float(
        result["shortage_rp10_30_mwh"] <= TOL
        and result["emergency_rp10_30_mwh"] <= TOL
    )
    for horizon in (10, 30, 60):
        result[f"feasible_la{horizon}"] = float(
            result[f"emergency_la{horizon}_mwh"] <= TOL
            and result[f"planned_emergency_la{horizon}_mw"] <= TOL
        )
    result["feasible_all"] = float(
        all(result[f"feasible_{design}"] > 0.5 for design in DESIGNS)
    )
    for design in DESIGNS:
        result[f"relief_rp10_vs_{design}"] = (
            result["loc_rp10"] - result[f"loc_{design}"]
        )
    return result


def _iqr(values: pd.Series) -> tuple[float, float]:
    if values.empty:
        return float("nan"), float("nan")
    q1, q3 = np.quantile(values.to_numpy(dtype=float), [0.25, 0.75])
    return float(q1), float(q3)


def _summary(cases: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (system, target), group in cases.groupby(
        ["system", "long_rmse_target"], sort=True
    ):
        common = group[group["feasible_all"] > 0.5]
        row: dict[str, Any] = {
            "system": system,
            "long_rmse_percent": int(round(100.0 * target)),
            "common_feasible": int(len(common)),
            "total_cases": int(len(group)),
        }
        for design in DESIGNS:
            row[f"feasible_{design}"] = int(
                (group[f"feasible_{design}"] > 0.5).sum()
            )
            q1, q3 = _iqr(common[f"loc_{design}"])
            row[f"loc_{design}_q1"] = q1
            row[f"loc_{design}_q3"] = q3
            q1, q3 = _iqr(common[f"relief_rp10_vs_{design}"])
            row[f"relief_{design}_q1"] = q1
            row[f"relief_{design}_q3"] = q3
        rows.append(row)
    return pd.DataFrame(rows)


def _money(value: float) -> str:
    if not np.isfinite(value):
        return "--"
    return f"{int(np.rint(value)):,}"


def _interval(row: pd.Series, prefix: str) -> str:
    return (
        f"[{_money(float(row[f'{prefix}_q1']))}, "
        f"{_money(float(row[f'{prefix}_q3']))}]"
    )


def _write_tables(summary: pd.DataFrame) -> None:
    lines = [
        "# Fair matched-horizon forecast comparison",
        "",
        "Each cell is a total-LOC IQR in dollars on the common feasible sample. "
        "The error column is the target RMSE at 30--60 minutes; 10-minute "
        "RMSE is the target divided by sqrt(2). Ramp requirements have no "
        "uncertainty proxy. Positive feasibility slack is treated as product "
        "infeasibility, not as a market purchase.",
        "",
        "## Total LOC IQR",
        "",
        "| System | Long-horizon RMSE | Common feasible | RP 10 | RP 10+30 | "
        "LA 10 | LA 30 | LA 60 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in summary.iterrows():
        lines.append(
            f"| {row['system']} | {int(row['long_rmse_percent'])}% | "
            f"{int(row['common_feasible'])}/{int(row['total_cases'])} | "
            f"{_interval(row, 'loc_rp10')} | "
            f"{_interval(row, 'loc_rp10_30')} | "
            f"{_interval(row, 'loc_la10')} | "
            f"{_interval(row, 'loc_la30')} | "
            f"{_interval(row, 'loc_la60')} |"
        )
    lines.extend(
        [
            "",
            "## Design-specific feasibility",
            "",
            "| System | Long-horizon RMSE | RP 10 | RP 10+30 | LA 10 | LA 30 | "
            "LA 60 |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for _, row in summary.iterrows():
        total = int(row["total_cases"])
        lines.append(
            f"| {row['system']} | {int(row['long_rmse_percent'])}% | "
            + " | ".join(
                f"{int(row[f'feasible_{design}'])}/{total}"
                for design in DESIGNS
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Paired relief IQR relative to RP 10",
            "",
            "Positive values mean the column design has lower LOC.",
            "",
            "| System | Long-horizon RMSE | RP 10+30 | LA 10 | LA 30 | LA 60 |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for _, row in summary.iterrows():
        lines.append(
            f"| {row['system']} | {int(row['long_rmse_percent'])}% | "
            f"{_interval(row, 'relief_rp10_30')} | "
            f"{_interval(row, 'relief_la10')} | "
            f"{_interval(row, 'relief_la30')} | "
            f"{_interval(row, 'relief_la60')} |"
        )
    lines.extend(
        [
            "",
            "RP 10+30 uses nested cumulative capability, so headroom, footroom, "
            "and ramp capability cannot be double-counted. The numerical "
            "product-shortage variable has a 3,000 dollars/MWh penalty and is "
            "used only to identify infeasible paths. This table has not been "
            "inserted into the manuscript.",
            "",
        ]
    )
    MARKDOWN_PATH.write_text("\n".join(lines))

    latex = [
        r"\begin{table*}[!t]",
        r"\caption{Fair matched-horizon forecast comparison. Entries are "
        r"total-LOC IQRs in dollars on the common feasible sample.}",
        r"\label{tab:fair_forecast_horizon}",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{2.8pt}",
        r"\begin{tabular}{llrrrrrr}",
        r"\toprule",
        r"System & RMSE & Feasible & RP 10 & RP 10+30 & LA 10 & LA 30 & LA 60 \\",
        r"\midrule",
    ]
    for _, row in summary.iterrows():
        fields = [
            str(row["system"]),
            f"{int(row['long_rmse_percent'])}\\%",
            f"{int(row['common_feasible'])}/{int(row['total_cases'])}",
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
        "--system", choices=("10gen", "RTS93", "both"), default="both"
    )
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if not (args.rts_root / "RTS_Data").exists():
        raise SystemExit(f"RTS-GMLC data not found at {args.rts_root}.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    profiles = load_rts_profiles(args.rts_root)
    controllable_profiles = load_rts_profiles(
        args.rts_root, subtract_hydro=False
    )
    systems = ("10gen", "RTS93") if args.system == "both" else (args.system,)
    settings = {
        "10gen": {
            "days": seasonal_days(),
            "profiles": profiles,
            "fleet": ten_generator_fleet(),
            "ramp_multiplier": 0.2,
        },
        "RTS93": {
            "days": external_check_days(),
            "profiles": controllable_profiles,
            "fleet": rts_controllable_fleet(args.rts_root),
            "ramp_multiplier": 0.6,
        },
    }
    if args.quick:
        for system in systems:
            settings[system]["days"] = settings[system]["days"][:1]

    existing = pd.read_csv(CASE_PATH) if CASE_PATH.exists() else pd.DataFrame()
    if args.force and not existing.empty:
        existing = existing[~existing["system"].isin(systems)].copy()
    rows = existing.to_dict("records") if not existing.empty else []
    completed = {_key(row) for row in rows}

    total_new = 0
    for system in systems:
        config = settings[system]
        for month, day in config["days"]:
            actual_raw, baseline_raw = config["profiles"][(month, day)]
            if system == "10gen":
                actual, _ = scale_profile(
                    actual_raw, baseline_raw, target_mean=1100.0
                )
            else:
                actual = actual_raw
            for target in ERROR_TARGETS:
                seeds = (0,) if target == 0.0 else NONZERO_SEEDS
                if args.quick and target > 0.0:
                    seeds = seeds[:1]
                for seed in seeds:
                    identity = (system, month, day, target, seed)
                    if identity in completed:
                        continue
                    # Four independent revision layers imply long-lead
                    # standard deviation 2 * innovation_sigma.
                    innovation_sigma = target / 2.0
                    forecast = make_forecast_matrix(
                        actual,
                        max_horizon=13,
                        sigma_rel=innovation_sigma,
                        seed=seed,
                        error_model="ar1_revision",
                        ar1_rho=0.9,
                    )
                    result = _run_path(
                        fleet=config["fleet"],
                        actual=actual,
                        forecast=forecast,
                        ramp_multiplier=float(config["ramp_multiplier"]),
                    )
                    rows.append(
                        {
                            "system": system,
                            "month": month,
                            "day": day,
                            "long_rmse_target": target,
                            "innovation_sigma_rel": innovation_sigma,
                            "seed": seed,
                            "forecast_model": "ar1_revision",
                            "forecast_rho": 0.9,
                            "ramp_multiplier": config["ramp_multiplier"],
                            "uncertainty_proxy_fraction": 0.0,
                            "product_feasibility_penalty": FEASIBILITY_PENALTY,
                            "realized_rmse_10min": _relative_rmse(
                                actual, forecast, 2
                            ),
                            "realized_rmse_30min": _relative_rmse(
                                actual, forecast, 6
                            ),
                            "realized_rmse_60min": _relative_rmse(
                                actual, forecast, 12
                            ),
                            **result,
                        }
                    )
                    completed.add(identity)
                    total_new += 1
                    pd.DataFrame(rows).to_csv(CASE_PATH, index=False)
                    print(
                        f"new={total_new}: {system} {month:02d}-{day:02d}, "
                        f"long RMSE={100*target:.0f}%, seed={seed}",
                        flush=True,
                    )

    cases = pd.DataFrame(rows)
    rmse_columns = (
        "realized_rmse_10min",
        "realized_rmse_30min",
        "realized_rmse_60min",
    )
    if any(column not in cases or cases[column].isna().any() for column in rmse_columns):
        profile_cache: dict[tuple[str, int, int], np.ndarray] = {}
        for index, row in cases.iterrows():
            if all(
                column in cases and pd.notna(row.get(column))
                for column in rmse_columns
            ):
                continue
            system = str(row["system"])
            month, day = int(row["month"]), int(row["day"])
            profile_key = (system, month, day)
            if profile_key not in profile_cache:
                actual_raw, baseline_raw = settings[system]["profiles"][
                    (month, day)
                ]
                if system == "10gen":
                    actual, _ = scale_profile(
                        actual_raw, baseline_raw, target_mean=1100.0
                    )
                else:
                    actual = actual_raw
                profile_cache[profile_key] = actual
            actual = profile_cache[profile_key]
            forecast = make_forecast_matrix(
                actual,
                max_horizon=13,
                sigma_rel=float(row["innovation_sigma_rel"]),
                seed=int(row["seed"]),
                error_model="ar1_revision",
                ar1_rho=0.9,
            )
            cases.loc[index, "realized_rmse_10min"] = _relative_rmse(
                actual, forecast, 2
            )
            cases.loc[index, "realized_rmse_30min"] = _relative_rmse(
                actual, forecast, 6
            )
            cases.loc[index, "realized_rmse_60min"] = _relative_rmse(
                actual, forecast, 12
            )
        cases.to_csv(CASE_PATH, index=False)
    selected = cases[cases["system"].isin(systems)].copy()
    summary = _summary(selected)
    summary.to_csv(SUMMARY_PATH, index=False)
    _write_tables(summary)
    metadata = {
        "case_rows": int(len(selected)),
        "forecast_model": "ar1_revision",
        "forecast_rho": 0.9,
        "long_rmse_targets": list(ERROR_TARGETS),
        "ten_minute_rmse_multiplier": float(1.0 / np.sqrt(2.0)),
        "uncertainty_proxy_fraction": 0.0,
        "product_feasibility_penalty": FEASIBILITY_PENALTY,
        "positive_product_slack_is_infeasible": True,
        "la_planned_emergency_is_infeasible": True,
        "product_intervals": {
            "rp10": [2],
            "rp10_30": [2, 6],
        },
        "la_horizons": {"la10": 3, "la30": 7, "la60": 13},
        "systems": {
            system: {
                "ramp_multiplier": settings[system]["ramp_multiplier"],
                "days": [
                    f"{month:02d}-{day:02d}"
                    for month, day in settings[system]["days"]
                ],
            }
            for system in systems
        },
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"wrote {SUMMARY_PATH}")
    print(f"wrote {MARKDOWN_PATH}")


if __name__ == "__main__":
    main()
