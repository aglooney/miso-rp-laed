"""Run the reviewer-motivated robustness and identification study.

Usage
-----
PYTHONPATH=analysis .venv/bin/python analysis/run_revision_study.py

The script writes checkpointed CSV files under ``outputs/revision``.  Existing
configuration rows are reused, so an interrupted run can be resumed.
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
    rts_thermal_fleet,
    scale_profile,
    seasonal_days,
    ten_generator_fleet,
)
from market_models import (
    generator_frame,
    make_forecast_matrix,
    simulate_day,
    summarize_simulation,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RTS = ROOT / "data" / "RTS-GMLC"
OUTPUT_DIR = ROOT / "outputs" / "revision"
AGGREGATE_PATH = OUTPUT_DIR / "aggregate_results.csv"
GENERATOR_PATH = OUTPUT_DIR / "generator_panel.csv"
METADATA_PATH = OUTPUT_DIR / "study_metadata.json"
SHORTAGE_PENALTY = 65.0


def _config_key(row: dict[str, Any]) -> str:
    fields = (
        "system",
        "study",
        "month",
        "day",
        "horizon",
        "ramp_multiplier",
        "upward_adder_fraction",
        "forecast_type",
        "forecast_model",
        "forecast_rho",
        "sigma_rel",
        "seed",
        "shortage_penalty",
    )
    values: list[str] = []
    for field in fields:
        value = row.get(field, "")
        if pd.isna(value):
            value = ""
        values.append(str(value))
    return "|".join(values)


def _load_existing(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def _save_rows(existing: pd.DataFrame, rows: list[dict[str, Any]], path: Path) -> pd.DataFrame:
    if not rows:
        return existing
    addition = pd.DataFrame(rows)
    combined = pd.concat([existing, addition], ignore_index=True)
    combined.to_csv(path, index=False)
    rows.clear()
    return combined


def _run_case(
    *,
    system: str,
    study: str,
    month: int,
    day: int,
    fleet,
    actual: np.ndarray,
    forecast_baseline: np.ndarray | None,
    forecast_type: str,
    horizon: int,
    ramp_multiplier: float,
    upward_adder_fraction: float,
    sigma_rel: float,
    seed: int,
    shortage_penalty: float,
    keep_generators: bool,
    forecast_model: str = "independent",
    forecast_rho: float = 0.0,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    scaled_fleet = fleet.with_ramp_multiplier(ramp_multiplier)
    forecast = make_forecast_matrix(
        actual,
        max_horizon=max(horizon, 3),
        day_ahead=forecast_baseline,
        sigma_rel=sigma_rel,
        seed=seed,
        error_model=forecast_model,
        ar1_rho=forecast_rho,
    )
    sim = simulate_day(
        fleet=scaled_fleet,
        actual_load=actual,
        forecast_matrix=forecast,
        horizon=horizon,
        upward_adder_fraction=upward_adder_fraction,
        downward_adder_fraction=0.0,
        shortage_penalty=shortage_penalty,
    )
    identifiers = {
        "system": system,
        "study": study,
        "month": month,
        "day": day,
        "horizon": horizon,
        "ramp_multiplier": ramp_multiplier,
        "upward_adder_fraction": upward_adder_fraction,
        "forecast_type": forecast_type,
        "forecast_model": forecast_model,
        "forecast_rho": forecast_rho,
        "sigma_rel": sigma_rel,
        "seed": seed,
        "shortage_penalty": shortage_penalty,
    }
    aggregate = {**identifiers, **summarize_simulation(sim)}
    aggregate["emergency_rp_mwh"] = aggregate["shed_rp_mwh"] + aggregate["spill_rp_mwh"]
    aggregate["emergency_la_mwh"] = aggregate["shed_la_mwh"] + aggregate["spill_la_mwh"]
    generator_rows: list[dict[str, Any]] = []
    if keep_generators:
        features = generator_frame(sim)
        for idx, name in enumerate(features["generator"]):
            row = {**identifiers, "generator": str(name)}
            for field, values in features.items():
                if field != "generator":
                    row[field] = float(values[idx])
            generator_rows.append(row)
    return aggregate, generator_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rts-root", type=Path, default=DEFAULT_RTS)
    parser.add_argument("--quick", action="store_true", help="Run a two-day smoke subset.")
    parser.add_argument(
        "--forecast-rho",
        type=float,
        default=0.9,
        help="AR(1) correlation for the controlled forecast-error study.",
    )
    parser.add_argument(
        "--refresh-forecast-error",
        action="store_true",
        help="Replace only the checkpointed controlled forecast-error cases.",
    )
    parser.add_argument(
        "--refresh-external93-forecast-error",
        action="store_true",
        help="Replace only the checkpointed RTS-93 controlled forecast-error cases.",
    )
    args = parser.parse_args()

    if not (args.rts_root / "RTS_Data").exists():
        raise SystemExit(
            "RTS-GMLC data not found. Clone https://github.com/GridMod/RTS-GMLC "
            f"to {args.rts_root}."
        )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    profiles = load_rts_profiles(args.rts_root)
    controllable_profiles = load_rts_profiles(
        args.rts_root, subtract_hydro=False
    )
    ten_fleet = ten_generator_fleet()
    rts_fleet = rts_thermal_fleet(args.rts_root)
    rts_controllable = rts_controllable_fleet(args.rts_root)

    aggregate = _load_existing(AGGREGATE_PATH)
    generator = _load_existing(GENERATOR_PATH)
    if args.refresh_forecast_error and not aggregate.empty:
        removed = int((aggregate["study"] == "forecast_error").sum())
        aggregate = aggregate[aggregate["study"] != "forecast_error"].copy()
        aggregate.to_csv(AGGREGATE_PATH, index=False)
        print(f"removed {removed} checkpointed forecast-error cases", flush=True)
    if args.refresh_external93_forecast_error and not aggregate.empty:
        removed = int(
            (aggregate["study"] == "external93_forecast_error").sum()
        )
        aggregate = aggregate[
            aggregate["study"] != "external93_forecast_error"
        ].copy()
        aggregate.to_csv(AGGREGATE_PATH, index=False)
        print(
            f"removed {removed} checkpointed RTS-93 forecast-error cases",
            flush=True,
        )
    completed = {_config_key(row) for row in aggregate.to_dict("records")}
    generator_completed = {_config_key(row) for row in generator.to_dict("records")}
    aggregate_buffer: list[dict[str, Any]] = []
    generator_buffer: list[dict[str, Any]] = []

    days = seasonal_days()
    external_days = external_check_days()
    if args.quick:
        days = [(1, 15), (7, 15)]
        external_days = [(1, 15), (7, 15)]

    configurations: list[dict[str, Any]] = []

    # Main deterministic factorial: ramp stress and ramp requirement.
    for month, day in days:
        for ramp_multiplier in (0.1, 0.2, 0.4, 0.8):
            for adder in (0.0, 0.08, 0.16, 0.24):
                configurations.append(
                    dict(
                        system="10gen",
                        study="ramp_requirement",
                        month=month,
                        day=day,
                        horizon=13,
                        ramp_multiplier=ramp_multiplier,
                        upward_adder_fraction=adder,
                        forecast_type="perfect",
                        sigma_rel=0.0,
                        seed=0,
                        shortage_penalty=SHORTAGE_PENALTY,
                        keep_generators=(
                            (ramp_multiplier, adder)
                            in ((0.1, 0.08), (0.1, 0.16), (0.4, 0.16))
                        ),
                    )
                )

    # Horizon ablation under the main severe deterministic setting.
    for month, day in days:
        for horizon in (1, 3, 7, 13):
            configurations.append(
                dict(
                    system="10gen",
                    study="horizon",
                    month=month,
                    day=day,
                    horizon=horizon,
                    ramp_multiplier=0.1,
                    upward_adder_fraction=0.08,
                    forecast_type="perfect",
                    sigma_rel=0.0,
                    seed=0,
                    shortage_penalty=SHORTAGE_PENALTY,
                    keep_generators=False,
                )
            )

    # Controlled forecast-error experiment with a persistent AR(1) delivery
    # shock, coherent rolling revisions, and common random numbers.
    for month, day in days:
        for sigma in (0.0, 0.01, 0.03, 0.05):
            seeds = (0,) if sigma == 0.0 else (11, 22, 33)
            for seed in seeds:
                configurations.append(
                    dict(
                        system="10gen",
                        study="forecast_error",
                        month=month,
                        day=day,
                        horizon=13,
                        ramp_multiplier=0.2,
                        upward_adder_fraction=0.08,
                        forecast_type="controlled_ar1",
                        forecast_model="ar1_delivery",
                        forecast_rho=args.forecast_rho,
                        sigma_rel=sigma,
                        seed=seed,
                        shortage_penalty=SHORTAGE_PENALTY,
                        keep_generators=False,
                    )
                )

    # Natural RTS day-ahead forecast errors as a deliberately severe check.
    for month, day in days:
        configurations.append(
            dict(
                system="10gen",
                study="natural_day_ahead",
                month=month,
                day=day,
                horizon=13,
                ramp_multiplier=0.4,
                upward_adder_fraction=0.08,
                forecast_type="rts_day_ahead",
                sigma_rel=0.0,
                seed=0,
                shortage_penalty=SHORTAGE_PENALTY,
                keep_generators=False,
            )
        )

    # Full-day, multi-season external check on the 73-unit convex RTS fleet.
    for month, day in external_days:
        for ramp_multiplier in (0.3, 0.6):
            configurations.append(
                dict(
                    system="RTS73",
                    study="external",
                    month=month,
                    day=day,
                    horizon=13,
                    ramp_multiplier=ramp_multiplier,
                    upward_adder_fraction=0.08,
                    forecast_type="perfect",
                    sigma_rel=0.0,
                    seed=0,
                    shortage_penalty=SHORTAGE_PENALTY,
                    keep_generators=True,
                )
            )

    # Multi-season check matching the accepted draft's 93-unit controllable
    # thermal-plus-hydro RTS reduction.
    for month, day in external_days:
        for ramp_multiplier in (0.2, 0.3, 0.4, 0.6, 0.8):
            configurations.append(
                dict(
                    system="RTS93",
                    study="external93",
                    month=month,
                    day=day,
                    horizon=13,
                    ramp_multiplier=ramp_multiplier,
                    upward_adder_fraction=0.16,
                    forecast_type="perfect",
                    sigma_rel=0.0,
                    seed=0,
                    shortage_penalty=SHORTAGE_PENALTY,
                    keep_generators=np.isclose(ramp_multiplier, 0.3),
                )
            )

    # Controlled forecast-error check on the 93-unit reduction, using the
    # same persistent delivery-time AR(1) paths and common-random-number
    # seeds as the 10-generator experiment.
    for month, day in external_days:
        for sigma in (0.0, 0.01, 0.03, 0.05):
            seeds = (0,) if sigma == 0.0 else (11, 22, 33)
            for seed in seeds:
                configurations.append(
                    dict(
                        system="RTS93",
                        study="external93_forecast_error",
                        month=month,
                        day=day,
                        horizon=13,
                        ramp_multiplier=0.6,
                        upward_adder_fraction=0.16,
                        forecast_type="controlled_ar1",
                        forecast_model="ar1_delivery",
                        forecast_rho=args.forecast_rho,
                        sigma_rel=sigma,
                        seed=seed,
                        shortage_penalty=SHORTAGE_PENALTY,
                        keep_generators=False,
                    )
                )

    # Feasible multi-day horizon ablation on the 93-unit reduction.
    for month, day in external_days:
        for horizon in (1, 3, 7, 13):
            configurations.append(
                dict(
                    system="RTS93",
                    study="external93_horizon",
                    month=month,
                    day=day,
                    horizon=horizon,
                    ramp_multiplier=0.3,
                    upward_adder_fraction=0.16,
                    forecast_type="perfect",
                    sigma_rel=0.0,
                    seed=0,
                    shortage_penalty=SHORTAGE_PENALTY,
                    keep_generators=False,
                )
            )

    total = len(configurations)
    new_count = 0
    for number, config in enumerate(configurations, start=1):
        key = _config_key(config)
        needs_aggregate = key not in completed
        needs_generators = config["keep_generators"] and key not in generator_completed
        if not needs_aggregate and not needs_generators:
            continue
        profile_source = (
            controllable_profiles if config["system"] == "RTS93" else profiles
        )
        actual_raw, forecast_raw = profile_source[(config["month"], config["day"])]
        if config["system"] == "10gen":
            actual, day_ahead = scale_profile(
                actual_raw, forecast_raw, target_mean=1100.0
            )
            fleet = ten_fleet
        elif config["system"] == "RTS73":
            actual, day_ahead = actual_raw, forecast_raw
            fleet = rts_fleet
        else:
            actual, day_ahead = actual_raw, forecast_raw
            fleet = rts_controllable
        baseline = (
            day_ahead if config["forecast_type"] == "rts_day_ahead" else None
        )
        row, gen_rows = _run_case(
            fleet=fleet,
            actual=actual,
            forecast_baseline=baseline,
            **config,
        )
        if needs_aggregate:
            aggregate_buffer.append(row)
            completed.add(key)
        if needs_generators:
            generator_buffer.extend(gen_rows)
            generator_completed.add(key)
        new_count += 1
        if new_count % 10 == 0:
            aggregate = _save_rows(aggregate, aggregate_buffer, AGGREGATE_PATH)
            generator = _save_rows(generator, generator_buffer, GENERATOR_PATH)
            print(f"completed {number}/{total}; new cases={new_count}", flush=True)

    aggregate = _save_rows(aggregate, aggregate_buffer, AGGREGATE_PATH)
    generator = _save_rows(generator, generator_buffer, GENERATOR_PATH)
    metadata = {
        "rts_commit": "3ece0d3725c844056132393ee252b3083dd4eab4",
        "interval_minutes": 5,
        "forecast_error_model": "ar1_delivery",
        "forecast_rho": args.forecast_rho,
        "shortage_penalty": SHORTAGE_PENALTY,
        "ten_generator_days": [f"{m:02d}-{d:02d}" for m, d in days],
        "external_days": [f"{m:02d}-{d:02d}" for m, d in external_days],
        "aggregate_cases": int(len(aggregate)),
        "generator_rows": int(len(generator)),
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
