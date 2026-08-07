"""Audit the mechanism-isolation forecast comparison."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs" / "forecast_horizon_fair"
CASE_PATH = OUTPUT_DIR / "fair_forecast_cases.csv"
SUMMARY_PATH = OUTPUT_DIR / "fair_forecast_iqr.csv"
METADATA_PATH = OUTPUT_DIR / "fair_forecast_metadata.json"

DESIGNS = ("rp10", "rp10_30", "la10", "la30", "la60")
KEYS = ["system", "month", "day", "long_rmse_target", "seed"]
TOL = 1e-6


def main() -> None:
    cases = pd.read_csv(CASE_PATH)
    summary = pd.read_csv(SUMMARY_PATH)
    metadata = json.loads(METADATA_PATH.read_text())

    assert len(cases) == 400
    assert cases.groupby("system").size().to_dict() == {
        "10gen": 280,
        "RTS93": 120,
    }
    assert not cases.duplicated(KEYS).any()
    assert set(cases["forecast_model"]) == {"ar1_revision"}
    assert np.allclose(cases["forecast_rho"], 0.9)
    assert np.allclose(
        cases["innovation_sigma_rel"] * 2.0,
        cases["long_rmse_target"],
    )
    assert np.allclose(cases["uncertainty_proxy_fraction"], 0.0)
    assert np.allclose(cases["product_feasibility_penalty"], 3000.0)

    for design in DESIGNS:
        assert float(cases[f"loc_{design}"].min()) >= -1e-4
        assert float(cases[f"emergency_{design}_mwh"].min()) >= -1e-9
    expected_rp10 = (
        (cases["shortage_rp10_mwh"] <= TOL)
        & (cases["emergency_rp10_mwh"] <= TOL)
    )
    expected_rp10_30 = (
        (cases["shortage_rp10_30_mwh"] <= TOL)
        & (cases["emergency_rp10_30_mwh"] <= TOL)
    )
    assert np.array_equal(
        expected_rp10.astype(float), cases["feasible_rp10"]
    )
    assert np.array_equal(
        expected_rp10_30.astype(float), cases["feasible_rp10_30"]
    )
    for horizon in (10, 30, 60):
        expected = (
            (cases[f"emergency_la{horizon}_mwh"] <= TOL)
            & (cases[f"planned_emergency_la{horizon}_mw"] <= TOL)
        )
        assert np.array_equal(
            expected.astype(float), cases[f"feasible_la{horizon}"]
        )
    expected_all = np.logical_and.reduce(
        [cases[f"feasible_{design}"] > 0.5 for design in DESIGNS]
    )
    assert np.array_equal(expected_all.astype(float), cases["feasible_all"])

    retained = cases[cases["feasible_all"] > 0.5]
    assert float(retained["shortage_rp10_mwh"].max()) <= TOL
    assert float(retained["shortage_rp10_30_mwh"].max()) <= TOL
    assert (
        float(
            retained[
                [
                    "max_product_price_rp10",
                    "max_product_price_rp10_30",
                ]
            ].max().max()
        )
        < 0.01 * 3000.0
    )

    # The finite-day realized RMSE should track the theoretical calibration.
    for target in (0.01, 0.03, 0.05):
        group = cases[np.isclose(cases["long_rmse_target"], target)]
        rmse10 = float(group["realized_rmse_10min"].median())
        rmse30 = float(group["realized_rmse_30min"].median())
        rmse60 = float(group["realized_rmse_60min"].median())
        assert 0.55 * target < rmse10 < 0.85 * target
        assert 0.90 * target < rmse30 < 1.20 * target
        assert 0.90 * target < rmse60 < 1.20 * target

    assert len(summary) == 8
    for _, row in summary.iterrows():
        group = cases[
            (cases["system"] == row["system"])
            & np.isclose(
                cases["long_rmse_target"],
                float(row["long_rmse_percent"]) / 100.0,
            )
        ]
        common = group[group["feasible_all"] > 0.5]
        assert int(row["common_feasible"]) == len(common)
        assert int(row["total_cases"]) == len(group)
        for design in DESIGNS:
            assert int(row[f"feasible_{design}"]) == int(
                (group[f"feasible_{design}"] > 0.5).sum()
            )
            expected = np.quantile(
                common[f"loc_{design}"], [0.25, 0.75]
            )
            reported = np.asarray(
                [row[f"loc_{design}_q1"], row[f"loc_{design}_q3"]]
            )
            assert np.allclose(expected, reported, atol=1e-9)
            expected = np.quantile(
                common[f"relief_rp10_vs_{design}"], [0.25, 0.75]
            )
            reported = np.asarray(
                [
                    row[f"relief_{design}_q1"],
                    row[f"relief_{design}_q3"],
                ]
            )
            assert np.allclose(expected, reported, atol=1e-9)

    assert metadata["case_rows"] == 400
    assert metadata["forecast_model"] == "ar1_revision"
    assert metadata["uncertainty_proxy_fraction"] == 0.0
    assert metadata["positive_product_slack_is_infeasible"] is True
    assert metadata["la_planned_emergency_is_infeasible"] is True
    print(
        "Fair forecast audit passed: 400 cases, calibrated errors, "
        "hard-equivalent feasibility, and 8 IQR rows verified."
    )


if __name__ == "__main__":
    main()
