"""Fast validation checks for the LP formulations and price extraction."""

from __future__ import annotations

import numpy as np

from market_models import (
    Fleet,
    make_forecast_matrix,
    simulate_day,
    solve_laed_step,
    solve_multi_ramp_product_step,
    solve_ramp_product_step,
)


def simple_fleet() -> Fleet:
    return Fleet(
        names=("cheap", "flex"),
        cost=np.asarray([20.0, 50.0]),
        pmax=np.asarray([100.0, 100.0]),
        ramp_up=np.asarray([10.0, 100.0]),
        ramp_down=np.asarray([10.0, 100.0]),
    )


def test_rp_balance_and_price_cap() -> None:
    fleet = simple_fleet()
    result = solve_ramp_product_step(
        load=80.0,
        forecast_10min=130.0,
        previous=np.asarray([70.0, 10.0]),
        fleet=fleet,
        upward_adder_mw=20.0,
        downward_adder_mw=0.0,
        shortage_penalty=65.0,
    )
    assert abs(np.sum(result["p"]) + result["shed"] - result["spill"] - 80.0) < 1e-7
    assert result["nu_up"] <= 65.0 + 1e-7
    assert np.all(result["p"] + result["ru"] <= fleet.pmax + 1e-7)


def test_laed_balance_and_tlmp_formula() -> None:
    fleet = simple_fleet()
    loads = np.asarray([80.0, 105.0, 130.0])
    result = solve_laed_step(
        forecast_load=loads,
        previous=np.asarray([70.0, 10.0]),
        fleet=fleet,
    )
    assert (
        np.max(
            np.abs(
                np.sum(result["p"], axis=1)
                + result["shed"]
                - result["spill"]
                - loads
            )
        )
        < 1e-7
    )
    expected = (
        result["lambda"][0]
        + result["mu_up"][1]
        - result["mu_down"][1]
        - result["mu_up"][0]
        + result["mu_down"][0]
    )
    assert np.max(np.abs(expected - result["tlmp_current"])) < 1e-9


def test_nested_10_and_30_minute_products() -> None:
    fleet = simple_fleet()
    previous = np.asarray([70.0, 10.0])
    result = solve_multi_ramp_product_step(
        load=80.0,
        product_forecasts=np.asarray([130.0, 170.0]),
        product_intervals=(2, 6),
        previous=previous,
        fleet=fleet,
        upward_adders_mw=(5.0, 10.0),
        downward_adders_mw=(0.0, 0.0),
        shortage_penalty=65.0,
    )

    assert abs(np.sum(result["p"]) + result["shed"] - result["spill"] - 80.0) < 1e-7
    assert np.all(
        np.sum(result["ru_cumulative"], axis=1) + result["short_up"]
        >= result["req_up"] - 1e-7
    )
    assert np.all(
        result["p"][None, :] + result["ru_cumulative"]
        <= fleet.pmax[None, :] + 1e-7
    )
    assert np.all(
        -result["p"][None, :] + result["rd_cumulative"] <= 1e-7
    )
    assert np.all(
        result["ru_segments"][0] <= 2.0 * fleet.ramp_up + 1e-7
    )
    assert np.all(
        result["ru_segments"][1] <= 4.0 * fleet.ramp_up + 1e-7
    )
    assert np.all(result["nu_up"] <= 65.0 + 1e-7)


def test_constant_load_has_zero_loc() -> None:
    fleet = Fleet(
        names=("one",),
        cost=np.asarray([25.0]),
        pmax=np.asarray([100.0]),
        ramp_up=np.asarray([100.0]),
        ramp_down=np.asarray([100.0]),
    )
    load = np.full(24, 50.0)
    forecast = make_forecast_matrix(load, max_horizon=5)
    sim = simulate_day(
        fleet=fleet,
        actual_load=load,
        forecast_matrix=forecast,
        horizon=5,
        upward_adder_fraction=0.0,
    )
    assert float(np.sum(sim["rp_loc"])) < 1e-6
    assert float(np.sum(sim["la_loc"])) < 1e-6
    assert float(np.sum(sim["tlmp_loc"])) < 1e-6


def test_constant_load_has_zero_loc_with_nested_products() -> None:
    fleet = Fleet(
        names=("one",),
        cost=np.asarray([25.0]),
        pmax=np.asarray([100.0]),
        ramp_up=np.asarray([100.0]),
        ramp_down=np.asarray([100.0]),
    )
    load = np.full(24, 50.0)
    forecast = make_forecast_matrix(load, max_horizon=13)
    sim = simulate_day(
        fleet=fleet,
        actual_load=load,
        forecast_matrix=forecast,
        horizon=7,
        upward_adder_fraction=(0.0, 0.0),
        product_intervals=(2, 6),
    )
    assert tuple(sim["rp_product_intervals"]) == (2, 6)
    assert sim["rp_ru_cumulative"].shape == (2, 1, 24)
    assert float(np.sum(sim["rp_loc"])) < 1e-6
    assert float(np.sum(sim["la_loc"])) < 1e-6
    assert float(np.sum(sim["tlmp_loc"])) < 1e-6


def test_ar1_forecast_is_coherent_and_reproducible() -> None:
    load = np.full(2_000, 1_000.0)
    kwargs = dict(
        max_horizon=13,
        sigma_rel=0.03,
        seed=299,
        error_model="ar1_delivery",
        ar1_rho=0.9,
    )
    forecast = make_forecast_matrix(load, **kwargs)
    repeated = make_forecast_matrix(load, **kwargs)
    different = make_forecast_matrix(load, **{**kwargs, "seed": 300})

    assert np.array_equal(forecast, repeated)
    assert not np.array_equal(forecast, different)
    assert np.array_equal(forecast[:, 0], load)

    # The same delivery interval keeps the same standardized shock as the
    # horizon rolls, while lead-time scaling makes the absolute error shrink.
    delivery = 100
    origins = np.arange(delivery - 4, delivery)
    leads = delivery - origins
    errors = forecast[origins, leads] - load[delivery]
    scales = 0.03 * load[delivery] * np.minimum(np.sqrt(leads), 2.0)
    standardized = errors / scales
    assert np.max(np.abs(standardized - standardized[0])) < 1e-12
    assert np.all(np.diff(np.abs(errors)) <= 1e-12)

    # Lead-one errors recover the latent delivery shocks, whose sample
    # first-order autocorrelation should be close to the configured rho.
    lead_one = (forecast[:-1, 1] - load[1:]) / (0.03 * load[1:])
    sample_rho = np.corrcoef(lead_one[:-1], lead_one[1:])[0, 1]
    assert 0.85 < sample_rho < 0.95


def test_ar1_revision_forecast_has_nested_revisions() -> None:
    load = np.full(4_000, 1_000.0)
    kwargs = dict(
        max_horizon=13,
        sigma_rel=0.025,
        seed=299,
        error_model="ar1_revision",
        ar1_rho=0.9,
    )
    forecast = make_forecast_matrix(load, **kwargs)
    repeated = make_forecast_matrix(load, **kwargs)
    assert np.array_equal(forecast, repeated)
    assert np.array_equal(forecast[:, 0], load)

    # Each rolling update for a fixed delivery removes one revision innovation.
    delivery = 100
    errors = np.asarray(
        [
            forecast[delivery - lead, lead] - load[delivery]
            for lead in (1, 2, 3, 4)
        ]
    )
    revision_increments = np.diff(np.r_[0.0, errors])
    assert np.all(np.abs(revision_increments) > 1e-9)

    # A 2.5% one-step innovation gives about 5% long-lead RMSE, capped
    # after four intervals.  Sample estimates are checked on a long series.
    start = 20
    stop = len(load) - 20
    rmses = {}
    for lead in (1, 2, 4, 12):
        origins = np.arange(start, stop)
        target = origins + lead
        error = forecast[origins, lead] - load[target]
        rmses[lead] = float(np.sqrt(np.mean(error**2)) / 1_000.0)
    assert 0.022 < rmses[1] < 0.028
    assert 0.032 < rmses[2] < 0.039
    assert 0.045 < rmses[4] < 0.055
    assert abs(rmses[12] - rmses[4]) < 0.002

    lead_one = (forecast[:-1, 1] - load[1:]) / (0.025 * load[1:])
    sample_rho = np.corrcoef(lead_one[:-1], lead_one[1:])[0, 1]
    assert 0.85 < sample_rho < 0.95


if __name__ == "__main__":
    test_rp_balance_and_price_cap()
    test_laed_balance_and_tlmp_formula()
    test_nested_10_and_30_minute_products()
    test_constant_load_has_zero_loc()
    test_constant_load_has_zero_loc_with_nested_products()
    test_ar1_forecast_is_coherent_and_reproducible()
    test_ar1_revision_forecast_has_nested_revisions()
    print("All market-model checks passed.")
