"""Data loaders for the 10-generator and RTS-GMLC study systems."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from market_models import Fleet


TEN_GEN_COST = np.asarray([185.0, 30.0, 55.0, 15.0, 20.0, 19.5, 48.0, 60.0, 57.0, 50.0])
TEN_GEN_CAPACITY = np.asarray([22.0, 170.0, 85.0, 230.0, 613.0, 686.0, 45.0, 50.0, 260.0, 400.0])
TEN_GEN_BASE_RAMP = np.asarray([75.0, 175.0, 75.0, 75.0, 375.0, 100.0, 100.0, 50.0, 50.0, 125.0])


def ten_generator_fleet() -> Fleet:
    return Fleet(
        names=tuple(str(i) for i in range(1, 11)),
        cost=TEN_GEN_COST.copy(),
        pmax=TEN_GEN_CAPACITY.copy(),
        ramp_up=TEN_GEN_BASE_RAMP.copy(),
        ramp_down=TEN_GEN_BASE_RAMP.copy(),
    )


def _read_total(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    index_cols = ["Year", "Month", "Day", "Period"]
    value_cols = [c for c in frame.columns if c not in index_cols]
    frame["value"] = frame[value_cols].sum(axis=1)
    return frame[index_cols + ["value"]]


def _merge_net_load(parts: list[tuple[pd.DataFrame, float]]) -> pd.DataFrame:
    index_cols = ["Year", "Month", "Day", "Period"]
    result = None
    for frame, sign in parts:
        component = frame.rename(columns={"value": f"value_{len(result.columns) if result is not None else 0}"})
        if result is None:
            result = component.copy()
            result["net_load"] = sign * result.iloc[:, -1]
        else:
            value_name = component.columns[-1]
            result = result.merge(component, on=index_cols, how="inner")
            result["net_load"] += sign * result[value_name]
    assert result is not None
    return result[index_cols + ["net_load"]]


def _interpolate_hourly(frame: pd.DataFrame) -> pd.DataFrame:
    """Linearly interpolate hourly day-ahead data to five-minute intervals."""
    records: list[dict[str, float | int]] = []
    for (year, month, day), group in frame.groupby(["Year", "Month", "Day"], sort=True):
        group = group.sort_values("Period")
        hour_x = (group["Period"].to_numpy(dtype=float) - 1.0) * 12.0
        values = group["value"].to_numpy(dtype=float)
        five_minute_x = np.arange(288, dtype=float)
        interpolated = np.interp(five_minute_x, hour_x, values)
        for period, value in enumerate(interpolated, start=1):
            records.append(
                {
                    "Year": int(year),
                    "Month": int(month),
                    "Day": int(day),
                    "Period": period,
                    "value": float(value),
                }
            )
    return pd.DataFrame.from_records(records)


def load_rts_profiles(
    rts_root: Path, *, subtract_hydro: bool = True
) -> dict[tuple[int, int], tuple[np.ndarray, np.ndarray]]:
    """Load realized and day-ahead single-bus net load for every RTS day.

    Set ``subtract_hydro=False`` when hydro and run-of-river generators are
    represented explicitly in the controllable fleet.
    """
    ts = rts_root / "RTS_Data" / "timeseries_data_files"
    actual_parts = [
        (_read_total(ts / "Load" / "REAL_TIME_regional_Load.csv"), 1.0),
        (_read_total(ts / "WIND" / "REAL_TIME_wind.csv"), -1.0),
        (_read_total(ts / "PV" / "REAL_TIME_pv.csv"), -1.0),
        (_read_total(ts / "RTPV" / "REAL_TIME_rtpv.csv"), -1.0),
    ]
    if subtract_hydro:
        actual_parts.append(
            (_read_total(ts / "Hydro" / "REAL_TIME_hydro.csv"), -1.0)
        )
    actual = _merge_net_load(actual_parts)

    forecast_parts = [
        (_interpolate_hourly(_read_total(ts / "Load" / "DAY_AHEAD_regional_Load.csv")), 1.0),
        (_interpolate_hourly(_read_total(ts / "WIND" / "DAY_AHEAD_wind.csv")), -1.0),
        (_interpolate_hourly(_read_total(ts / "PV" / "DAY_AHEAD_pv.csv")), -1.0),
        (_interpolate_hourly(_read_total(ts / "RTPV" / "DAY_AHEAD_rtpv.csv")), -1.0),
    ]
    if subtract_hydro:
        forecast_parts.append(
            (
                _interpolate_hourly(
                    _read_total(ts / "Hydro" / "DAY_AHEAD_hydro.csv")
                ),
                -1.0,
            )
        )
    forecast = _merge_net_load(forecast_parts)
    merged = actual.merge(
        forecast,
        on=["Year", "Month", "Day", "Period"],
        suffixes=("_actual", "_forecast"),
        how="inner",
    )
    profiles: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}
    for (month, day), group in merged.groupby(["Month", "Day"], sort=True):
        group = group.sort_values("Period")
        if len(group) != 288:
            continue
        actual_path = np.maximum(group["net_load_actual"].to_numpy(dtype=float), 0.0)
        forecast_path = np.maximum(group["net_load_forecast"].to_numpy(dtype=float), 0.0)
        profiles[(int(month), int(day))] = (actual_path, forecast_path)
    return profiles


def scale_profile(
    actual: np.ndarray, forecast: np.ndarray, *, target_mean: float
) -> tuple[np.ndarray, np.ndarray]:
    scale = float(target_mean) / float(np.mean(actual))
    return actual * scale, forecast * scale


def rts_thermal_fleet(rts_root: Path) -> Fleet:
    """Return the 73-unit convex thermal fleet from RTS-GMLC.

    Variable renewables and scheduled hydro are represented in net load.
    Minimum-output and commitment constraints are intentionally relaxed to
    match the paper's convex economic-dispatch setting.
    """
    gen_path = rts_root / "RTS_Data" / "SourceData" / "gen.csv"
    frame = pd.read_csv(gen_path)
    frame = frame[frame["Unit Type"].isin(["CC", "CT", "NUCLEAR", "STEAM"])].copy()
    heat_rate_cols = ["HR_avg_0", "HR_incr_1", "HR_incr_2", "HR_incr_3"]
    heat_rates = frame[heat_rate_cols].apply(pd.to_numeric, errors="coerce")
    mean_heat_rate = heat_rates.mean(axis=1)
    cost = (
        frame["Fuel Price $/MMBTU"].to_numpy(dtype=float)
        * mean_heat_rate.to_numpy(dtype=float)
        / 1000.0
        + frame["VOM"].to_numpy(dtype=float)
    )
    ramp_5min = frame["Ramp Rate MW/Min"].to_numpy(dtype=float) * 5.0
    return Fleet(
        names=tuple(frame["GEN UID"].astype(str)),
        cost=np.asarray(cost, dtype=float),
        pmax=frame["PMax MW"].to_numpy(dtype=float),
        ramp_up=ramp_5min,
        ramp_down=ramp_5min,
    )


def rts_controllable_fleet(rts_root: Path) -> Fleet:
    """Return the 93-unit convex thermal and hydro RTS-GMLC fleet.

    Wind, utility PV, and rooftop PV are represented in net load. Hydro and
    run-of-river units are dispatchable here, matching the accepted draft's
    93-generator controllable reduction. Minimum-output, commitment, water,
    and network constraints remain relaxed.
    """
    gen_path = rts_root / "RTS_Data" / "SourceData" / "gen.csv"
    frame = pd.read_csv(gen_path)
    frame = frame[
        frame["Unit Type"].isin(["CC", "CT", "NUCLEAR", "STEAM", "HYDRO", "ROR"])
    ].copy()
    heat_rate_cols = ["HR_avg_0", "HR_incr_1", "HR_incr_2", "HR_incr_3"]
    heat_rates = frame[heat_rate_cols].apply(pd.to_numeric, errors="coerce")
    mean_heat_rate = heat_rates.mean(axis=1)
    cost = (
        frame["Fuel Price $/MMBTU"].to_numpy(dtype=float)
        * mean_heat_rate.to_numpy(dtype=float)
        / 1000.0
        + frame["VOM"].to_numpy(dtype=float)
    )
    ramp_5min = frame["Ramp Rate MW/Min"].to_numpy(dtype=float) * 5.0
    fleet = Fleet(
        names=tuple(frame["GEN UID"].astype(str)),
        cost=np.asarray(cost, dtype=float),
        pmax=frame["PMax MW"].to_numpy(dtype=float),
        ramp_up=ramp_5min,
        ramp_down=ramp_5min,
    )
    if fleet.n != 93:
        raise AssertionError(f"Expected 93 controllable RTS units, found {fleet.n}.")
    return fleet


def seasonal_days() -> list[tuple[int, int]]:
    """Seven complete days centered in each representative season."""
    return [(month, day) for month in (1, 4, 7, 10) for day in range(12, 19)]


def external_check_days() -> list[tuple[int, int]]:
    """Three full days per season for the larger RTS thermal check."""
    return [(month, day) for month in (1, 4, 7, 10) for day in (13, 15, 17)]
