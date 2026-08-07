"""Audit the generated data against the quantitative claims in main.tex."""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs" / "revision"


def main() -> None:
    aggregate = pd.read_csv(OUTPUT / "aggregate_results.csv")
    generator = pd.read_csv(OUTPUT / "generator_panel.csv")
    summary = json.loads((OUTPUT / "publication_summary.json").read_text())
    metadata = json.loads((OUTPUT / "study_metadata.json").read_text())

    assert len(aggregate) == 1120
    assert len(generator) == 3708
    assert metadata["aggregate_cases"] == len(aggregate)
    assert metadata["generator_rows"] == len(generator)
    assert metadata["rts_commit"] == "3ece0d3725c844056132393ee252b3083dd4eab4"
    assert metadata["forecast_error_model"] == "ar1_delivery"
    assert np.isclose(metadata["forecast_rho"], 0.9)
    assert np.isclose(metadata["shortage_penalty"], 65.0)
    assert np.allclose(aggregate["shortage_penalty"], 65.0)
    assert np.allclose(generator["shortage_penalty"], 65.0)
    assert aggregate.groupby("study").size().to_dict() == {
        "external": 24,
        "external93": 60,
        "external93_forecast_error": 120,
        "external93_horizon": 48,
        "forecast_error": 280,
        "horizon": 112,
        "natural_day_ahead": 28,
        "ramp_requirement": 448,
    }
    assert len(aggregate[aggregate["system"].isin(["10gen", "RTS93"])]) == 1096

    loc_columns = ["loc_rp", "loc_rp_energy", "loc_la", "loc_tlmp"]
    assert (aggregate[loc_columns] >= -1e-5).all().all()
    assert float(np.max(np.abs(aggregate["loc_tlmp"]))) < 1e-5

    featured = aggregate[
        (aggregate["month"] == 10)
        & (aggregate["day"] == 15)
        & (aggregate["study"] == "ramp_requirement")
        & np.isclose(aggregate["ramp_multiplier"], 0.1)
        & np.isclose(aggregate["upward_adder_fraction"], 0.16)
    ].iloc[0]
    assert featured["emergency_rp_mwh"] < 1e-7
    assert featured["emergency_la_mwh"] < 1e-7
    assert np.isclose(featured["loc_rp_energy"], summary["feature_loc_rp_energy"])
    assert np.isclose(featured["loc_rp"], summary["feature_loc_rp"])
    assert np.isclose(featured["loc_la"], summary["feature_loc_la"])
    assert np.isclose(featured["loc_tlmp"], summary["feature_loc_tlmp"])
    assert np.isclose(featured["max_rcup"], 65.0)

    regression = summary["regression"]
    assert regression["model"]["n"] == 1256
    assert regression["model"]["clusters"] == 103
    assert regression["Bind frequency"]["LOC/MW_p"] < 0.001
    assert regression["Bind frequency"]["RP-LA relief/MW_p"] < 0.001

    ramp = aggregate[
        (aggregate["study"] == "ramp_requirement")
        & np.isclose(aggregate["upward_adder_fraction"], 0.16)
    ].copy()
    ramp["clean"] = (
        (ramp["emergency_rp_mwh"] <= 1e-6)
        & (ramp["emergency_la_mwh"] <= 1e-6)
    )
    expected_ramp = {
        0.1: (2, 1.0, 0.0, 3154.208594689472, 312.83932017063535,
              1693.893486337928, 2841.369274518837, 3988.8450626997455),
        0.2: (6, 2.0 / 3.0, 1.0 / 3.0, 620.4738300773652, 40.45735157780973,
              24.61428410050576, 149.26389103829655, 412.82884315920967),
        0.4: (14, 5.0 / 7.0, 2.0 / 7.0, 75.63344566784417, 19.922530997650806,
              1.2413557299651075, 22.745424525743147, 79.18870980663196),
        0.8: (20, 0.15, 0.85, 0.0, 0.0, 0.0, 0.0, 0.0),
    }
    for multiplier, (
        count, share, ties, rp_median, la_median, q1, relief_median, q3
    ) in expected_ramp.items():
        group = ramp[np.isclose(ramp["ramp_multiplier"], multiplier)]
        clean = group[group["clean"]]
        delta = clean["loc_rp"] - clean["loc_la"]
        assert len(clean) == count
        assert np.isclose(np.mean(delta > 1e-6), share)
        assert np.isclose(np.mean(np.abs(delta) <= 1e-6), ties)
        assert np.isclose(np.median(clean["loc_rp"]), rp_median)
        assert np.isclose(np.median(clean["loc_la"]), la_median)
        assert np.allclose(np.quantile(delta, [0.25, 0.5, 0.75]),
                           [q1, relief_median, q3])

    natural = aggregate[aggregate["study"] == "natural_day_ahead"].copy()
    assert len(natural) == 28
    assert np.all(natural["loc_rp"] < natural["loc_la"] - 1e-6)
    natural["clean"] = (
        (natural["emergency_rp_mwh"] <= 1e-6)
        & (natural["emergency_la_mwh"] <= 1e-6)
    )
    natural_clean = natural[natural["clean"]]
    assert len(natural_clean) == 12
    assert np.all(natural_clean["loc_rp"] < natural_clean["loc_la"] - 1e-6)
    assert np.isclose(np.median(natural_clean["loc_rp"]), 10.019577552539431)
    assert np.isclose(np.median(natural_clean["loc_la"]), 52996.94351967699)

    controlled = aggregate[aggregate["study"] == "forecast_error"]
    assert set(controlled["forecast_model"]) == {"ar1_delivery"}
    assert np.allclose(controlled["forecast_rho"], 0.9)
    assert set(controlled["forecast_type"]) == {"controlled_ar1"}
    expected_controlled = {
        0.00: (6, 1, 2.0 / 3.0, 620.4738300773652, 40.45735157780973,
               24.61428410050576, 149.26389103829655, 412.82884315920967),
        0.01: (18, 3, 0.5, 620.4738300773652, 254.32812114748398,
               -33.35369526321608, 49.88217525661822, 562.587328900443),
        0.03: (19, 3, 2.0 / 19.0, 1113.447660154785, 2245.266106007341,
               -2829.227810620622, -1162.1347679347439, -425.9850632482992),
        0.05: (19, 3, 0.0, 1432.3326146390034, 11641.449317261216,
               -18184.831301687693, -10602.869710130291, -8854.687185006454),
    }
    for sigma, (
        count, seeds, share, rp_median, la_median, q1, relief_median, q3
    ) in expected_controlled.items():
        group = controlled[np.isclose(controlled["sigma_rel"], sigma)]
        clean = group[
            (group["emergency_rp_mwh"] <= 1e-6)
            & (group["emergency_la_mwh"] <= 1e-6)
        ]
        delta = clean["loc_rp"] - clean["loc_la"]
        assert len(group) == 28 * seeds
        assert len(clean) == count
        assert np.isclose(np.mean(delta > 1e-6), share)
        assert np.isclose(np.median(clean["loc_rp"]), rp_median)
        assert np.isclose(np.median(clean["loc_la"]), la_median)
        assert np.allclose(np.quantile(delta, [0.25, 0.5, 0.75]),
                           [q1, relief_median, q3])

    external93 = aggregate[aggregate["study"] == "external93"]
    assert len(external93) == 60
    assert np.all(external93["emergency_rp_mwh"] <= 1e-6)
    assert np.all(external93["emergency_la_mwh"] <= 1e-6)
    assert np.all(external93["loc_rp"] > external93["loc_la"] + 1e-6)
    expected_external93 = {
        0.2: (77688.66858759066, 15315.68321238915,
              52313.18905433032, 62199.36414970124, 78328.71385297782),
        0.3: (50633.99157737284, 6061.769190049641,
              30789.735870776676, 43892.941681161596, 61660.00034903418),
        0.4: (37004.862218175294, 4932.392276800794,
              18144.477867721784, 31980.51764995885, 46440.91499118281),
        0.6: (17386.249866961596, 1634.796669223284,
              5499.820221010812, 15448.421565644177, 23072.9894019849),
        0.8: (10226.054509868123, 613.1204077985304,
              3754.254135483751, 8330.070550129876, 13714.995404344048),
    }
    for multiplier, (rp_median, la_median, q1, relief_median, q3) in (
        expected_external93.items()
    ):
        group = external93[np.isclose(external93["ramp_multiplier"], multiplier)]
        delta = group["loc_rp"] - group["loc_la"]
        assert len(group) == 12
        assert np.isclose(np.median(group["loc_rp"]), rp_median)
        assert np.isclose(np.median(group["loc_la"]), la_median)
        assert np.allclose(np.quantile(delta, [0.25, 0.5, 0.75]),
                           [q1, relief_median, q3])

    external93_controlled = aggregate[
        aggregate["study"] == "external93_forecast_error"
    ]
    assert set(external93_controlled["forecast_model"]) == {"ar1_delivery"}
    assert np.allclose(external93_controlled["forecast_rho"], 0.9)
    assert set(external93_controlled["forecast_type"]) == {"controlled_ar1"}
    expected_external93_controlled = {
        0.00: (12, 1, 1.0, 17386.249866961596, 1634.796669223284,
               5499.820221010811, 15448.421565644177, 23072.9894019849),
        0.01: (36, 3, 8.0 / 9.0, 17386.249866961596, 5390.044002653178,
               1464.5182898674234, 12164.390494767998, 19122.265221765505),
        0.03: (36, 3, 11.0 / 36.0, 17386.249866961596, 28462.981193444608,
               -20659.05744851262, -10797.175553970184, 1280.1150417151048),
        0.05: (36, 3, 2.0 / 36.0, 21475.21772788697, 61194.57673672185,
               -73344.64820784215, -48164.418097335925, -24478.173286156987),
    }
    for sigma, (
        count, seeds, share, rp_median, la_median, q1, relief_median, q3
    ) in (
        expected_external93_controlled.items()
    ):
        group = external93_controlled[
            np.isclose(external93_controlled["sigma_rel"], sigma)
        ]
        clean = group[
            (group["emergency_rp_mwh"] <= 1e-6)
            & (group["emergency_la_mwh"] <= 1e-6)
        ]
        delta = clean["loc_rp"] - clean["loc_la"]
        assert len(group) == 12 * seeds
        assert len(clean) == count
        assert np.isclose(np.mean(delta > 1e-6), share)
        assert np.isclose(np.median(clean["loc_rp"]), rp_median)
        assert np.isclose(np.median(clean["loc_la"]), la_median)
        assert np.allclose(np.quantile(delta, [0.25, 0.5, 0.75]),
                           [q1, relief_median, q3])
    external93_zero = external93_controlled[
        np.isclose(external93_controlled["sigma_rel"], 0.0)
    ].sort_values(["month", "day"])
    external93_ramp_six = external93[
        np.isclose(external93["ramp_multiplier"], 0.6)
    ].sort_values(["month", "day"])
    assert np.allclose(external93_zero["loc_rp"], external93_ramp_six["loc_rp"])
    assert np.allclose(external93_zero["loc_la"], external93_ramp_six["loc_la"])

    horizon = aggregate[aggregate["study"] == "external93_horizon"].copy()
    horizon["clean"] = (
        (horizon["emergency_rp_mwh"] <= 1e-6)
        & (horizon["emergency_la_mwh"] <= 1e-6)
    )
    complete_days = horizon.groupby(["month", "day"])["clean"].all()
    complete_days = set(complete_days[complete_days].index)
    assert len(complete_days) == 10
    complete = horizon[
        horizon.apply(lambda row: (row["month"], row["day"]) in complete_days, axis=1)
    ]
    expected_horizon_medians = {
        1: (50633.99157737284, 50633.99157737284),
        3: (50633.99157737284, 23876.095942848984),
        7: (50633.99157737284, 11811.777234423964),
        13: (50633.99157737284, 5405.023908091394),
    }
    for lookahead, (rp_median, la_median) in expected_horizon_medians.items():
        group = complete[complete["horizon"] == lookahead]
        assert np.isclose(np.median(group["loc_rp"]), rp_median)
        assert np.isclose(np.median(group["loc_la"]), la_median)

    manuscript = (ROOT / "main_revised.tex").read_text()
    assert re.search(r"\b\d+(?:\.\d+)?k\b", manuscript) is None
    assert r"\usepackage{cite}" in manuscript
    assert "Raw day-ahead" not in manuscript
    assert "MISO2026RCPUncertainty" in manuscript
    assert r"with $LOC^{LA}-LOC^{RP}$ on the right axis" in manuscript
    for token in (
        "1\\% error & 18/84 & 50.0\\% & 50 & [$-33$, 563]",
        "3\\% error & 19/84 & 10.5\\% & $-1{,}162$ & [$-2{,}829$, $-426$]",
        "5\\% error & 19/84 & 0.0\\% & $-10{,}603$ & [$-18{,}185$, $-8{,}855$]",
        "Ramp 0.2 & 12/12 & 100.0\\% & 62{,}199 & [52{,}313, 78{,}329]",
        "Ramp 0.4 & 12/12 & 100.0\\% & 31{,}981 & [18{,}144, 46{,}441]",
        "Ramp 0.8 & 12/12 & 100.0\\% & 8{,}330 & [3{,}754, 13{,}715]",
        "1\\% error & 36/36 & 88.9\\% & 12{,}164 & [1{,}465, 19{,}122]",
        "3\\% error & 36/36 & 30.6\\% & $-10{,}797$ & [$-20{,}659$, 1{,}280]",
        "5\\% error & 36/36 & 5.6\\% & $-48{,}164$ & [$-73{,}345$, $-24{,}478$]",
    ):
        assert token in manuscript

    print(
        "Publication audit passed: "
        f"{len(aggregate)} cases, {len(generator)} generator rows, "
        f"max |TLMP LOC|={np.max(np.abs(aggregate['loc_tlmp'])):.3g}."
    )


if __name__ == "__main__":
    main()
