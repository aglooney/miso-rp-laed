"""Audit the standalone matched-horizon forecast-error experiment."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs" / "forecast_horizon"
CASE_PATH = OUTPUT_DIR / "forecast_horizon_cases.csv"
SUMMARY_PATH = OUTPUT_DIR / "forecast_horizon_iqr.csv"
METADATA_PATH = OUTPUT_DIR / "forecast_horizon_metadata.json"
REVISION_PATH = ROOT / "outputs" / "revision" / "aggregate_results.csv"

DESIGNS = ("rp10", "rp10_30", "la10", "la30", "la60")
KEYS = ["system", "month", "day", "sigma_rel", "seed"]


def main() -> None:
    cases = pd.read_csv(CASE_PATH)
    summary = pd.read_csv(SUMMARY_PATH)
    revision = pd.read_csv(REVISION_PATH)
    metadata = json.loads(METADATA_PATH.read_text())

    assert len(cases) == 400
    assert cases.groupby("system").size().to_dict() == {
        "10gen": 280,
        "RTS93": 120,
    }
    assert not cases.duplicated(KEYS).any()
    assert set(cases["forecast_model"]) == {"ar1_delivery"}
    assert np.allclose(cases["forecast_rho"], 0.9)
    assert set(cases["sigma_rel"]) == {0.0, 0.01, 0.03, 0.05}

    for design in DESIGNS:
        assert float(cases[f"loc_{design}"].min()) >= -1e-4
        assert float(cases[f"emergency_{design}_mwh"].min()) >= -1e-9
    clean = np.logical_and.reduce(
        [cases[f"emergency_{design}_mwh"] <= 1e-6 for design in DESIGNS]
    )
    assert np.array_equal(clean.astype(float), cases["clean_all"].to_numpy())

    references = []
    for system, study in (
        ("10gen", "forecast_error"),
        ("RTS93", "external93_forecast_error"),
    ):
        subset = revision[
            (revision["system"] == system) & (revision["study"] == study)
        ][
            KEYS
            + [
                "loc_rp",
                "loc_la",
                "emergency_rp_mwh",
                "emergency_la_mwh",
            ]
        ]
        references.append(subset)
    reference = pd.concat(references, ignore_index=True)
    merged = cases.merge(
        reference,
        on=KEYS,
        how="left",
        validate="one_to_one",
    )
    assert not merged["loc_rp"].isna().any()
    assert np.allclose(merged["loc_rp10"], merged["loc_rp"], atol=1e-6)
    assert np.allclose(merged["loc_la60"], merged["loc_la"], atol=1e-6)
    assert np.allclose(
        merged["emergency_rp10_mwh"],
        merged["emergency_rp_mwh"],
        atol=1e-9,
    )
    assert np.allclose(
        merged["emergency_la60_mwh"],
        merged["emergency_la_mwh"],
        atol=1e-9,
    )

    assert len(summary) == 8
    for _, row in summary.iterrows():
        group = cases[
            (cases["system"] == row["system"])
            & np.isclose(
                cases["sigma_rel"], float(row["error_percent"]) / 100.0
            )
        ]
        common = group[group["clean_all"] > 0.5]
        assert int(row["clean_cases"]) == len(common)
        assert int(row["total_cases"]) == len(group)
        for design in DESIGNS:
            expected_loc = np.quantile(
                common[f"loc_{design}"], [0.25, 0.75]
            )
            reported_loc = np.asarray(
                [
                    row[f"loc_{design}_q1"],
                    row[f"loc_{design}_q3"],
                ]
            )
            assert np.allclose(expected_loc, reported_loc, atol=1e-9)
            expected_relief = np.quantile(
                common[f"relief_rp10_vs_{design}"], [0.25, 0.75]
            )
            reported_relief = np.asarray(
                [
                    row[f"relief_{design}_q1"],
                    row[f"relief_{design}_q3"],
                ]
            )
            assert np.allclose(expected_relief, reported_relief, atol=1e-9)

    assert metadata["case_rows"] == 400
    assert metadata["common_random_numbers"] is True
    assert metadata["designs"]["rp10_30"]["product_intervals"] == [2, 6]
    assert metadata["designs"]["rp10_30"]["nested_cumulative_awards"] is True
    print(
        "Forecast-horizon audit passed: 400 cases, 8 summary rows, "
        "common-sample IQRs verified."
    )


if __name__ == "__main__":
    main()
