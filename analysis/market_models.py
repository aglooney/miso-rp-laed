"""Convex rolling-horizon market models used by the revision study.

The implementation deliberately uses scipy.optimize.linprog (HiGHS) so every
reported result can be reproduced without a commercial solver.  All power
quantities are MW, costs and prices are $/MWh, and each dispatch interval is
five minutes unless ``dt_hours`` is changed.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np
from scipy.optimize import linprog


TOL = 1e-7


@dataclass(frozen=True)
class Fleet:
    names: tuple[str, ...]
    cost: np.ndarray
    pmax: np.ndarray
    ramp_up: np.ndarray
    ramp_down: np.ndarray

    def __post_init__(self) -> None:
        arrays = (self.cost, self.pmax, self.ramp_up, self.ramp_down)
        if not all(len(a) == len(self.names) for a in arrays):
            raise ValueError("Fleet arrays must have the same length.")
        if np.any(self.pmax <= 0) or np.any(self.ramp_up <= 0) or np.any(self.ramp_down <= 0):
            raise ValueError("Capacity and ramp limits must be strictly positive.")

    @property
    def n(self) -> int:
        return len(self.names)

    def with_ramp_multiplier(self, multiplier: float) -> "Fleet":
        if multiplier <= 0:
            raise ValueError("Ramp multiplier must be positive.")
        return replace(
            self,
            ramp_up=np.asarray(self.ramp_up, dtype=float) * multiplier,
            ramp_down=np.asarray(self.ramp_down, dtype=float) * multiplier,
        )


def economic_initial_dispatch(load: float, fleet: Fleet) -> np.ndarray:
    """Return a zero-minimum, merit-order initial dispatch."""
    p = np.zeros(fleet.n)
    remaining = max(float(load), 0.0)
    for g in np.argsort(fleet.cost):
        award = min(float(fleet.pmax[g]), remaining)
        p[g] = award
        remaining -= award
        if remaining <= TOL:
            break
    return p


def _solve_lp(
    c: np.ndarray,
    *,
    a_ub: np.ndarray | None = None,
    b_ub: np.ndarray | None = None,
    a_eq: np.ndarray | None = None,
    b_eq: np.ndarray | None = None,
    bounds: list[tuple[float | None, float | None]] | None = None,
    label: str,
) -> Any:
    result = linprog(
        c,
        A_ub=a_ub,
        b_ub=b_ub,
        A_eq=a_eq,
        b_eq=b_eq,
        bounds=bounds,
        method="highs",
    )
    if not result.success:
        raise RuntimeError(f"{label} failed: {result.message}")
    return result


def _normalize_product_intervals(
    product_intervals: int | tuple[int, ...],
) -> tuple[int, ...]:
    if isinstance(product_intervals, (int, np.integer)):
        intervals = (int(product_intervals),)
    else:
        intervals = tuple(int(value) for value in product_intervals)
    if not intervals or any(value <= 0 for value in intervals):
        raise ValueError("Ramp-product intervals must be positive.")
    if tuple(sorted(set(intervals))) != intervals:
        raise ValueError("Ramp-product intervals must be strictly increasing.")
    return intervals


def _normalize_product_values(
    values: float | tuple[float, ...],
    *,
    count: int,
    label: str,
) -> tuple[float, ...]:
    if np.isscalar(values):
        normalized = (float(values),) * count
    else:
        normalized = tuple(float(value) for value in values)
    if len(normalized) != count:
        raise ValueError(f"{label} must have one value per ramp product.")
    return normalized


def solve_multi_ramp_product_step(
    *,
    load: float,
    product_forecasts: np.ndarray,
    product_intervals: tuple[int, ...],
    previous: np.ndarray,
    fleet: Fleet,
    upward_adders_mw: float | tuple[float, ...],
    downward_adders_mw: float | tuple[float, ...],
    shortage_penalty: float,
    voll: float = 3500.0,
) -> dict[str, Any]:
    """Clear energy with nested ramp products at one or more horizons.

    Awards are represented as nonnegative capability segments.  For product
    ``j``, the cumulative award includes segments 0 through ``j``.  Segment 0
    must be deliverable by the first product horizon; each later segment has
    only the incremental ramp time between adjacent horizons.  Cumulative
    headroom and footroom constraints prevent physical capability from being
    double-counted across products.
    """
    intervals = _normalize_product_intervals(product_intervals)
    product_count = len(intervals)
    forecasts = np.asarray(product_forecasts, dtype=float)
    if forecasts.shape != (product_count,):
        raise ValueError("product_forecasts must have one value per ramp product.")
    up_adders = _normalize_product_values(
        upward_adders_mw, count=product_count, label="upward_adders_mw"
    )
    down_adders = _normalize_product_values(
        downward_adders_mw, count=product_count, label="downward_adders_mw"
    )

    g_count = fleet.n
    p0 = 0
    ru0 = g_count
    rd0 = ru0 + product_count * g_count
    shed_i = rd0 + product_count * g_count
    spill_i = shed_i + 1
    zu0 = spill_i + 1
    zd0 = zu0 + product_count
    n_var = zd0 + product_count

    def ru_index(product: int, generator: int) -> int:
        return ru0 + product * g_count + generator

    def rd_index(product: int, generator: int) -> int:
        return rd0 + product * g_count + generator

    objective = np.zeros(n_var)
    objective[p0:ru0] = fleet.cost
    objective[shed_i] = voll
    objective[spill_i] = voll
    objective[zu0:zd0] = shortage_penalty
    objective[zd0:] = shortage_penalty

    req_up = np.asarray(
        [
            max(0.0, forecasts[j] - float(load)) + max(0.0, up_adders[j])
            for j in range(product_count)
        ]
    )
    req_down = np.asarray(
        [
            max(0.0, float(load) - forecasts[j]) + max(0.0, down_adders[j])
            for j in range(product_count)
        ]
    )

    rows: list[np.ndarray] = []
    rhs: list[float] = []

    # Current energy must remain feasible from the previously implemented point.
    for g in range(g_count):
        row = np.zeros(n_var)
        row[p0 + g] = 1.0
        rows.append(row)
        rhs.append(float(previous[g] + fleet.ramp_up[g]))

        row = np.zeros(n_var)
        row[p0 + g] = -1.0
        rows.append(row)
        rhs.append(float(fleet.ramp_down[g] - previous[g]))

    # Each cumulative product must fit within physical headroom and footroom.
    for g in range(g_count):
        for j in range(product_count):
            row = np.zeros(n_var)
            row[p0 + g] = 1.0
            for segment in range(j + 1):
                row[ru_index(segment, g)] = 1.0
            rows.append(row)
            rhs.append(float(fleet.pmax[g]))

            row = np.zeros(n_var)
            row[p0 + g] = -1.0
            for segment in range(j + 1):
                row[rd_index(segment, g)] = 1.0
            rows.append(row)
            rhs.append(0.0)

    up_req_rows = np.zeros(product_count, dtype=int)
    down_req_rows = np.zeros(product_count, dtype=int)
    for j in range(product_count):
        up_req_rows[j] = len(rows)
        row = np.zeros(n_var)
        for segment in range(j + 1):
            start = ru0 + segment * g_count
            row[start : start + g_count] = -1.0
        row[zu0 + j] = -1.0
        rows.append(row)
        rhs.append(-float(req_up[j]))

        down_req_rows[j] = len(rows)
        row = np.zeros(n_var)
        for segment in range(j + 1):
            start = rd0 + segment * g_count
            row[start : start + g_count] = -1.0
        row[zd0 + j] = -1.0
        rows.append(row)
        rhs.append(-float(req_down[j]))

    balance = np.zeros((1, n_var))
    balance[0, p0:ru0] = 1.0
    balance[0, shed_i] = 1.0
    balance[0, spill_i] = -1.0

    bounds: list[tuple[float | None, float | None]] = []
    bounds.extend((0.0, float(fleet.pmax[g])) for g in range(g_count))
    prior_interval = 0
    for interval in intervals:
        incremental_intervals = interval - prior_interval
        bounds.extend(
            (0.0, float(incremental_intervals * fleet.ramp_up[g]))
            for g in range(g_count)
        )
        prior_interval = interval
    prior_interval = 0
    for interval in intervals:
        incremental_intervals = interval - prior_interval
        bounds.extend(
            (0.0, float(incremental_intervals * fleet.ramp_down[g]))
            for g in range(g_count)
        )
        prior_interval = interval
    bounds.extend(
        [
            (0.0, max(float(load), 0.0)),
            (0.0, float(np.sum(fleet.pmax))),
        ]
    )
    bounds.extend((0.0, float(value)) for value in req_up)
    bounds.extend((0.0, float(value)) for value in req_down)

    result = _solve_lp(
        objective,
        a_ub=np.asarray(rows),
        b_ub=np.asarray(rhs),
        a_eq=balance,
        b_eq=np.asarray([load]),
        bounds=bounds,
        label="multi-horizon ramp-product clearing",
    )

    ru_segments = np.asarray(
        result.x[ru0:rd0], dtype=float
    ).reshape(product_count, g_count)
    rd_segments = np.asarray(
        result.x[rd0:shed_i], dtype=float
    ).reshape(product_count, g_count)
    up_prices = np.maximum(
        0.0, -np.asarray(result.ineqlin.marginals)[up_req_rows]
    )
    down_prices = np.maximum(
        0.0, -np.asarray(result.ineqlin.marginals)[down_req_rows]
    )
    return {
        "p": np.asarray(result.x[p0:ru0]),
        "ru_segments": ru_segments,
        "rd_segments": rd_segments,
        "ru_cumulative": np.cumsum(ru_segments, axis=0),
        "rd_cumulative": np.cumsum(rd_segments, axis=0),
        "shed": float(result.x[shed_i]),
        "spill": float(result.x[spill_i]),
        "short_up": np.asarray(result.x[zu0:zd0]),
        "short_down": np.asarray(result.x[zd0:]),
        "lambda": float(result.eqlin.marginals[0]),
        "nu_up": up_prices,
        "nu_down": down_prices,
        "req_up": req_up,
        "req_down": req_down,
        "product_intervals": intervals,
        "objective": float(result.fun),
    }


def solve_ramp_product_step(
    *,
    load: float,
    forecast_10min: float,
    previous: np.ndarray,
    fleet: Fleet,
    upward_adder_mw: float,
    downward_adder_mw: float,
    shortage_penalty: float,
    product_intervals: int = 2,
    voll: float = 3500.0,
) -> dict[str, Any]:
    """Clear current energy with co-optimized up/down ramp capability."""
    result = solve_multi_ramp_product_step(
        load=load,
        product_forecasts=np.asarray([forecast_10min], dtype=float),
        product_intervals=(product_intervals,),
        previous=previous,
        fleet=fleet,
        upward_adders_mw=(upward_adder_mw,),
        downward_adders_mw=(downward_adder_mw,),
        shortage_penalty=shortage_penalty,
        voll=voll,
    )
    return {
        "p": result["p"],
        "ru": result["ru_cumulative"][0],
        "rd": result["rd_cumulative"][0],
        "shed": result["shed"],
        "spill": result["spill"],
        "short_up": float(result["short_up"][0]),
        "short_down": float(result["short_down"][0]),
        "lambda": result["lambda"],
        "nu_up": float(result["nu_up"][0]),
        "nu_down": float(result["nu_down"][0]),
        "req_up": float(result["req_up"][0]),
        "req_down": float(result["req_down"][0]),
        "objective": result["objective"],
    }


def solve_laed_step(
    *,
    forecast_load: np.ndarray,
    previous: np.ndarray,
    fleet: Fleet,
    voll: float = 3500.0,
) -> dict[str, Any]:
    """Clear a finite-horizon LAED and return current LMP and TLMP."""
    forecast_load = np.asarray(forecast_load, dtype=float)
    horizon = len(forecast_load)
    g_count = fleet.n
    p_count = horizon * g_count
    shed0 = p_count
    spill0 = p_count + horizon
    n_var = p_count + 2 * horizon

    def p_index(k: int, g: int) -> int:
        return k * g_count + g

    objective = np.zeros(n_var)
    for k in range(horizon):
        objective[k * g_count : (k + 1) * g_count] = fleet.cost
        objective[shed0 + k] = voll
        objective[spill0 + k] = voll

    balance = np.zeros((horizon, n_var))
    for k in range(horizon):
        balance[k, k * g_count : (k + 1) * g_count] = 1.0
        balance[k, shed0 + k] = 1.0
        balance[k, spill0 + k] = -1.0

    rows: list[np.ndarray] = []
    rhs: list[float] = []
    up_rows = np.zeros((horizon, g_count), dtype=int)
    down_rows = np.zeros((horizon, g_count), dtype=int)
    for k in range(horizon):
        for g in range(g_count):
            up_rows[k, g] = len(rows)
            row = np.zeros(n_var)
            row[p_index(k, g)] = 1.0
            if k == 0:
                bound = float(previous[g] + fleet.ramp_up[g])
            else:
                row[p_index(k - 1, g)] = -1.0
                bound = float(fleet.ramp_up[g])
            rows.append(row)
            rhs.append(bound)

            down_rows[k, g] = len(rows)
            row = np.zeros(n_var)
            row[p_index(k, g)] = -1.0
            if k == 0:
                bound = float(fleet.ramp_down[g] - previous[g])
            else:
                row[p_index(k - 1, g)] = 1.0
                bound = float(fleet.ramp_down[g])
            rows.append(row)
            rhs.append(bound)

    bounds: list[tuple[float | None, float | None]] = []
    for _k in range(horizon):
        bounds.extend((0.0, float(fleet.pmax[g])) for g in range(g_count))
    bounds.extend((0.0, max(float(forecast_load[k]), 0.0)) for k in range(horizon))
    bounds.extend((0.0, float(np.sum(fleet.pmax))) for _k in range(horizon))

    result = _solve_lp(
        objective,
        a_ub=np.asarray(rows),
        b_ub=np.asarray(rhs),
        a_eq=balance,
        b_eq=forecast_load,
        bounds=bounds,
        label="LAED clearing",
    )
    p = np.asarray(result.x[:p_count]).reshape(horizon, g_count)
    lmp = np.asarray(result.eqlin.marginals)
    mu_up = -np.asarray(result.ineqlin.marginals)[up_rows]
    mu_down = -np.asarray(result.ineqlin.marginals)[down_rows]

    next_adjustment = (
        mu_up[1] - mu_down[1] if horizon > 1 else np.zeros(g_count)
    )
    current_adjustment = mu_up[0] - mu_down[0]
    tlmp_current = lmp[0] + next_adjustment - current_adjustment
    return {
        "p": p,
        "shed": np.asarray(result.x[shed0:spill0]),
        "spill": np.asarray(result.x[spill0:]),
        "lambda": lmp,
        "mu_up": mu_up,
        "mu_down": mu_down,
        "tlmp_current": tlmp_current,
        "objective": float(result.fun),
    }


def make_forecast_matrix(
    actual: np.ndarray,
    *,
    max_horizon: int,
    day_ahead: np.ndarray | None = None,
    sigma_rel: float = 0.0,
    seed: int = 0,
    error_model: str = "independent",
    ar1_rho: float = 0.9,
) -> np.ndarray:
    """Create rolling forecasts with observed current load and noisy futures.

    If ``day_ahead`` is supplied it is the baseline future forecast.
    ``ar1_delivery`` retains the legacy construction in which one AR(1) shock
    per delivery interval is scaled by lead time.  ``ar1_revision`` instead
    builds each forecast error from nested revision innovations: as delivery
    approaches, one innovation is removed from the error.  Each innovation
    layer is AR(1) along the delivery timeline, and independent layers produce
    standard deviation proportional to the square root of lead time, capped
    after four intervals.  Identical matrices can be reused across designs for
    paired comparisons.
    """
    actual = np.asarray(actual, dtype=float)
    baseline = actual if day_ahead is None else np.asarray(day_ahead, dtype=float)
    if len(actual) != len(baseline):
        raise ValueError("Actual and day-ahead paths must have equal length.")
    if error_model not in {"independent", "ar1_delivery", "ar1_revision"}:
        raise ValueError(
            "error_model must be 'independent', 'ar1_delivery', or "
            "'ar1_revision'."
        )
    if not -1.0 < ar1_rho < 1.0:
        raise ValueError("ar1_rho must be strictly between -1 and 1.")

    rng = np.random.default_rng(seed)
    delivery_shock = np.zeros(len(actual))
    if error_model == "ar1_delivery" and sigma_rel > 0.0:
        delivery_shock[0] = rng.normal()
        innovation_scale = np.sqrt(1.0 - ar1_rho**2)
        for idx in range(1, len(delivery_shock)):
            delivery_shock[idx] = (
                ar1_rho * delivery_shock[idx - 1]
                + innovation_scale * rng.normal()
            )

    revision_layers = np.zeros((4, len(actual)))
    if error_model == "ar1_revision" and sigma_rel > 0.0:
        innovation_scale = np.sqrt(1.0 - ar1_rho**2)
        for layer in range(len(revision_layers)):
            revision_layers[layer, 0] = rng.normal()
            for idx in range(1, len(actual)):
                revision_layers[layer, idx] = (
                    ar1_rho * revision_layers[layer, idx - 1]
                    + innovation_scale * rng.normal()
                )

    matrix = np.zeros((len(actual), max_horizon))
    for t in range(len(actual)):
        matrix[t, 0] = actual[t]
        for lead in range(1, max_horizon):
            idx = min(t + lead, len(actual) - 1)
            scale = sigma_rel * max(actual[idx], 1.0) * min(np.sqrt(lead), 2.0)
            if error_model == "ar1_delivery":
                error = scale * delivery_shock[idx]
            elif error_model == "ar1_revision":
                layer_count = min(lead, len(revision_layers))
                error = (
                    sigma_rel
                    * max(actual[idx], 1.0)
                    * float(np.sum(revision_layers[:layer_count, idx]))
                )
            else:
                error = rng.normal(0.0, scale)
            matrix[t, lead] = max(0.0, baseline[idx] + error)
    return matrix


def _best_response_energy(
    *,
    prices: np.ndarray,
    initial: np.ndarray,
    fleet: Fleet,
    dt_hours: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Solve each generator's intertemporal energy-only best response."""
    prices = np.asarray(prices, dtype=float)
    if prices.ndim == 1:
        prices = np.repeat(prices[None, :], fleet.n, axis=0)
    time_count = prices.shape[1]
    best_profit = np.zeros(fleet.n)
    schedules = np.zeros((fleet.n, time_count))
    for g in range(fleet.n):
        objective = -(prices[g] - fleet.cost[g]) * dt_hours
        rows: list[np.ndarray] = []
        rhs: list[float] = []
        for t in range(time_count):
            row = np.zeros(time_count)
            row[t] = 1.0
            if t == 0:
                bound = float(initial[g] + fleet.ramp_up[g])
            else:
                row[t - 1] = -1.0
                bound = float(fleet.ramp_up[g])
            rows.append(row)
            rhs.append(bound)

            row = np.zeros(time_count)
            row[t] = -1.0
            if t == 0:
                bound = float(fleet.ramp_down[g] - initial[g])
            else:
                row[t - 1] = 1.0
                bound = float(fleet.ramp_down[g])
            rows.append(row)
            rhs.append(bound)
        result = _solve_lp(
            objective,
            a_ub=np.asarray(rows),
            b_ub=np.asarray(rhs),
            bounds=[(0.0, float(fleet.pmax[g]))] * time_count,
            label=f"energy best response for {fleet.names[g]}",
        )
        schedules[g] = result.x
        best_profit[g] = -float(result.fun)
    return best_profit, schedules


def _best_response_multi_ramp_product(
    *,
    energy_prices: np.ndarray,
    up_prices: np.ndarray,
    down_prices: np.ndarray,
    initial: np.ndarray,
    fleet: Fleet,
    product_intervals: tuple[int, ...],
    dt_hours: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Solve each generator's joint energy/nested-ramp-product best response."""
    intervals = _normalize_product_intervals(product_intervals)
    time_count = len(energy_prices)
    product_count = len(intervals)
    up_prices = np.asarray(up_prices, dtype=float)
    down_prices = np.asarray(down_prices, dtype=float)
    if up_prices.shape != (product_count, time_count):
        raise ValueError("up_prices must be product-by-time.")
    if down_prices.shape != (product_count, time_count):
        raise ValueError("down_prices must be product-by-time.")

    best_profit = np.zeros(fleet.n)
    schedules = np.zeros((fleet.n, time_count))
    for g in range(fleet.n):
        q0 = 0
        u0 = time_count
        d0 = u0 + product_count * time_count
        objective = np.zeros(time_count + 2 * product_count * time_count)
        objective[q0:u0] = -(energy_prices - fleet.cost[g]) * dt_hours
        # Segment i contributes to every cumulative product j >= i.
        for segment in range(product_count):
            up_segment_price = np.sum(up_prices[segment:], axis=0)
            down_segment_price = np.sum(down_prices[segment:], axis=0)
            start = u0 + segment * time_count
            objective[start : start + time_count] = -up_segment_price * dt_hours
            start = d0 + segment * time_count
            objective[start : start + time_count] = -down_segment_price * dt_hours

        rows: list[np.ndarray] = []
        rhs: list[float] = []
        for t in range(time_count):
            row = np.zeros_like(objective)
            row[q0 + t] = 1.0
            if t == 0:
                bound = float(initial[g] + fleet.ramp_up[g])
            else:
                row[q0 + t - 1] = -1.0
                bound = float(fleet.ramp_up[g])
            rows.append(row)
            rhs.append(bound)

            row = np.zeros_like(objective)
            row[q0 + t] = -1.0
            if t == 0:
                bound = float(fleet.ramp_down[g] - initial[g])
            else:
                row[q0 + t - 1] = 1.0
                bound = float(fleet.ramp_down[g])
            rows.append(row)
            rhs.append(bound)

            for product in range(product_count):
                row = np.zeros_like(objective)
                row[q0 + t] = 1.0
                for segment in range(product + 1):
                    row[u0 + segment * time_count + t] = 1.0
                rows.append(row)
                rhs.append(float(fleet.pmax[g]))

                row = np.zeros_like(objective)
                row[q0 + t] = -1.0
                for segment in range(product + 1):
                    row[d0 + segment * time_count + t] = 1.0
                rows.append(row)
                rhs.append(0.0)

        bounds: list[tuple[float | None, float | None]] = []
        bounds.extend([(0.0, float(fleet.pmax[g]))] * time_count)
        prior_interval = 0
        for interval in intervals:
            incremental_intervals = interval - prior_interval
            bounds.extend(
                [(0.0, float(incremental_intervals * fleet.ramp_up[g]))]
                * time_count
            )
            prior_interval = interval
        prior_interval = 0
        for interval in intervals:
            incremental_intervals = interval - prior_interval
            bounds.extend(
                [(0.0, float(incremental_intervals * fleet.ramp_down[g]))]
                * time_count
            )
            prior_interval = interval
        result = _solve_lp(
            objective,
            a_ub=np.asarray(rows),
            b_ub=np.asarray(rhs),
            bounds=bounds,
            label=f"ramp-product best response for {fleet.names[g]}",
        )
        schedules[g] = result.x[q0:u0]
        best_profit[g] = -float(result.fun)
    return best_profit, schedules


def _best_response_ramp_product(
    *,
    energy_prices: np.ndarray,
    up_prices: np.ndarray,
    down_prices: np.ndarray,
    initial: np.ndarray,
    fleet: Fleet,
    product_intervals: int,
    dt_hours: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Backward-compatible one-product best response."""
    return _best_response_multi_ramp_product(
        energy_prices=energy_prices,
        up_prices=np.asarray(up_prices, dtype=float)[None, :],
        down_prices=np.asarray(down_prices, dtype=float)[None, :],
        initial=initial,
        fleet=fleet,
        product_intervals=(product_intervals,),
        dt_hours=dt_hours,
    )


def simulate_day(
    *,
    fleet: Fleet,
    actual_load: np.ndarray,
    forecast_matrix: np.ndarray,
    horizon: int,
    upward_adder_fraction: float | tuple[float, ...],
    downward_adder_fraction: float | tuple[float, ...] = 0.0,
    shortage_penalty: float = 65.0,
    product_intervals: int | tuple[int, ...] = 2,
    dt_hours: float = 1.0 / 12.0,
    voll: float = 3500.0,
) -> dict[str, Any]:
    """Run paired RP and LAED rolling simulations and compute LOC."""
    actual_load = np.asarray(actual_load, dtype=float)
    time_count = len(actual_load)
    product_horizons = _normalize_product_intervals(product_intervals)
    product_count = len(product_horizons)
    if (
        forecast_matrix.shape[0] != time_count
        or forecast_matrix.shape[1]
        < max(horizon, max(product_horizons) + 1)
    ):
        raise ValueError("Forecast matrix is too small for the requested horizons.")

    initial = economic_initial_dispatch(actual_load[0], fleet)
    rp_previous = initial.copy()
    la_previous = initial.copy()
    mean_load = float(np.mean(actual_load))
    up_adder_fractions = _normalize_product_values(
        upward_adder_fraction,
        count=product_count,
        label="upward_adder_fraction",
    )
    down_adder_fractions = _normalize_product_values(
        downward_adder_fraction,
        count=product_count,
        label="downward_adder_fraction",
    )
    up_adders = tuple(value * mean_load for value in up_adder_fractions)
    down_adders = tuple(value * mean_load for value in down_adder_fractions)

    rp_p = np.zeros((fleet.n, time_count))
    rp_ru_segments = np.zeros((product_count, fleet.n, time_count))
    rp_rd_segments = np.zeros_like(rp_ru_segments)
    rp_ru_cumulative = np.zeros_like(rp_ru_segments)
    rp_rd_cumulative = np.zeros_like(rp_ru_segments)
    rp_lambda = np.zeros(time_count)
    rp_nu_up_products = np.zeros((product_count, time_count))
    rp_nu_down_products = np.zeros_like(rp_nu_up_products)
    rp_shed = np.zeros(time_count)
    rp_spill = np.zeros(time_count)
    rp_short_up_products = np.zeros((product_count, time_count))
    rp_short_down_products = np.zeros_like(rp_short_up_products)
    rp_req_up_products = np.zeros_like(rp_short_up_products)
    rp_req_down_products = np.zeros_like(rp_short_up_products)

    la_p = np.zeros((fleet.n, time_count))
    la_lambda = np.zeros(time_count)
    la_tlmp = np.zeros((fleet.n, time_count))
    la_shed = np.zeros(time_count)
    la_spill = np.zeros(time_count)
    la_planned_shed_max = np.zeros(time_count)
    la_planned_spill_max = np.zeros(time_count)

    rp_bind_up = np.zeros((fleet.n, time_count), dtype=bool)
    rp_bind_down = np.zeros_like(rp_bind_up)
    la_bind_up = np.zeros_like(rp_bind_up)
    la_bind_down = np.zeros_like(rp_bind_up)

    for t in range(time_count):
        prior_rp = rp_previous.copy()
        rp = solve_multi_ramp_product_step(
            load=float(actual_load[t]),
            product_forecasts=np.asarray(
                [forecast_matrix[t, lead] for lead in product_horizons],
                dtype=float,
            ),
            product_intervals=product_horizons,
            previous=prior_rp,
            fleet=fleet,
            upward_adders_mw=up_adders,
            downward_adders_mw=down_adders,
            shortage_penalty=shortage_penalty,
            voll=voll,
        )
        rp_p[:, t] = rp["p"]
        rp_ru_segments[:, :, t] = rp["ru_segments"]
        rp_rd_segments[:, :, t] = rp["rd_segments"]
        rp_ru_cumulative[:, :, t] = rp["ru_cumulative"]
        rp_rd_cumulative[:, :, t] = rp["rd_cumulative"]
        rp_lambda[t] = rp["lambda"]
        rp_nu_up_products[:, t] = rp["nu_up"]
        rp_nu_down_products[:, t] = rp["nu_down"]
        rp_shed[t] = rp["shed"]
        rp_spill[t] = rp["spill"]
        rp_short_up_products[:, t] = rp["short_up"]
        rp_short_down_products[:, t] = rp["short_down"]
        rp_req_up_products[:, t] = rp["req_up"]
        rp_req_down_products[:, t] = rp["req_down"]
        rp_delta = rp["p"] - prior_rp
        bind_tol = 1e-5 + 1e-5 * np.maximum(fleet.ramp_up, fleet.ramp_down)
        rp_bind_up[:, t] = np.abs(rp_delta - fleet.ramp_up) <= bind_tol
        rp_bind_down[:, t] = np.abs(rp_delta + fleet.ramp_down) <= bind_tol
        rp_previous = rp["p"]

        prior_la = la_previous.copy()
        h = min(horizon, time_count - t)
        la = solve_laed_step(
            forecast_load=forecast_matrix[t, :h],
            previous=prior_la,
            fleet=fleet,
            voll=voll,
        )
        la_p[:, t] = la["p"][0]
        la_lambda[t] = la["lambda"][0]
        la_tlmp[:, t] = la["tlmp_current"]
        la_shed[t] = la["shed"][0]
        la_spill[t] = la["spill"][0]
        la_planned_shed_max[t] = float(np.max(la["shed"]))
        la_planned_spill_max[t] = float(np.max(la["spill"]))
        la_delta = la["p"][0] - prior_la
        la_bind_up[:, t] = np.abs(la_delta - fleet.ramp_up) <= bind_tol
        la_bind_down[:, t] = np.abs(la_delta + fleet.ramp_down) <= bind_tol
        la_previous = la["p"][0]

    rp_best, _ = _best_response_multi_ramp_product(
        energy_prices=rp_lambda,
        up_prices=rp_nu_up_products,
        down_prices=rp_nu_down_products,
        initial=initial,
        fleet=fleet,
        product_intervals=product_horizons,
        dt_hours=dt_hours,
    )
    rp_profit = np.sum(
        (rp_lambda[None, :] - fleet.cost[:, None]) * rp_p,
        axis=1,
    ) * dt_hours
    rp_profit += (
        np.sum(
            rp_nu_up_products[:, None, :] * rp_ru_cumulative
            + rp_nu_down_products[:, None, :] * rp_rd_cumulative,
            axis=(0, 2),
        )
        * dt_hours
    )
    rp_loc = rp_best - rp_profit

    rp_energy_best, _ = _best_response_energy(
        prices=rp_lambda, initial=initial, fleet=fleet, dt_hours=dt_hours
    )
    rp_energy_profit = (
        np.sum((rp_lambda[None, :] - fleet.cost[:, None]) * rp_p, axis=1) * dt_hours
    )
    rp_energy_loc = rp_energy_best - rp_energy_profit

    la_best, _ = _best_response_energy(
        prices=la_lambda, initial=initial, fleet=fleet, dt_hours=dt_hours
    )
    la_profit = (
        np.sum((la_lambda[None, :] - fleet.cost[:, None]) * la_p, axis=1) * dt_hours
    )
    la_loc = la_best - la_profit

    tlmp_best, _ = _best_response_energy(
        prices=la_tlmp, initial=initial, fleet=fleet, dt_hours=dt_hours
    )
    tlmp_profit = (
        np.sum((la_tlmp - fleet.cost[:, None]) * la_p, axis=1) * dt_hours
    )
    tlmp_loc = tlmp_best - tlmp_profit

    for label, loc in {
        "RP": rp_loc,
        "RP energy-only": rp_energy_loc,
        "LA-LMP": la_loc,
        "TLMP": tlmp_loc,
    }.items():
        if float(np.min(loc)) < -1e-4:
            raise AssertionError(f"{label} LOC is negative beyond tolerance: {np.min(loc)}")
        loc[np.abs(loc) < 1e-7] = 0.0

    return {
        "fleet": fleet,
        "initial": initial,
        "actual_load": actual_load,
        "rp_p": rp_p,
        # Backward-compatible aliases refer to the shortest-horizon product.
        "rp_ru": rp_ru_cumulative[0],
        "rp_rd": rp_rd_cumulative[0],
        "rp_ru_segments": rp_ru_segments,
        "rp_rd_segments": rp_rd_segments,
        "rp_ru_cumulative": rp_ru_cumulative,
        "rp_rd_cumulative": rp_rd_cumulative,
        "rp_product_intervals": np.asarray(product_horizons, dtype=int),
        "rp_lambda": rp_lambda,
        "rp_nu_up": rp_nu_up_products[0],
        "rp_nu_down": rp_nu_down_products[0],
        "rp_nu_up_products": rp_nu_up_products,
        "rp_nu_down_products": rp_nu_down_products,
        "rp_shed": rp_shed,
        "rp_spill": rp_spill,
        "rp_short_up": rp_short_up_products[0],
        "rp_short_down": rp_short_down_products[0],
        "rp_short_up_products": rp_short_up_products,
        "rp_short_down_products": rp_short_down_products,
        "rp_req_up": rp_req_up_products[0],
        "rp_req_down": rp_req_down_products[0],
        "rp_req_up_products": rp_req_up_products,
        "rp_req_down_products": rp_req_down_products,
        "rp_bind_up": rp_bind_up,
        "rp_bind_down": rp_bind_down,
        "rp_loc": rp_loc,
        "rp_energy_loc": rp_energy_loc,
        "rp_profit": rp_profit,
        "la_p": la_p,
        "la_lambda": la_lambda,
        "la_tlmp": la_tlmp,
        "la_shed": la_shed,
        "la_spill": la_spill,
        "la_planned_shed_max": la_planned_shed_max,
        "la_planned_spill_max": la_planned_spill_max,
        "la_bind_up": la_bind_up,
        "la_bind_down": la_bind_down,
        "la_loc": la_loc,
        "la_profit": la_profit,
        "tlmp_loc": tlmp_loc,
        "tlmp_profit": tlmp_profit,
        "dt_hours": dt_hours,
    }


def summarize_simulation(sim: dict[str, Any]) -> dict[str, float]:
    """Return publication-facing aggregate outcomes."""
    dt = float(sim["dt_hours"])
    return {
        "loc_rp": float(np.sum(sim["rp_loc"])),
        "loc_rp_energy": float(np.sum(sim["rp_energy_loc"])),
        "loc_la": float(np.sum(sim["la_loc"])),
        "loc_tlmp": float(np.sum(sim["tlmp_loc"])),
        "delta_la_rp": float(np.sum(sim["la_loc"]) - np.sum(sim["rp_loc"])),
        "shed_rp_mwh": float(np.sum(sim["rp_shed"]) * dt),
        "shed_la_mwh": float(np.sum(sim["la_shed"]) * dt),
        "spill_rp_mwh": float(np.sum(sim["rp_spill"]) * dt),
        "spill_la_mwh": float(np.sum(sim["la_spill"]) * dt),
        "short_up_mwh": float(np.sum(sim["rp_short_up"]) * dt),
        "short_down_mwh": float(np.sum(sim["rp_short_down"]) * dt),
        "rcup_active_share": float(np.mean(sim["rp_nu_up"] > 1e-7)),
        "rcdn_active_share": float(np.mean(sim["rp_nu_down"] > 1e-7)),
        "max_rcup": float(np.max(sim["rp_nu_up"])),
        "max_tlmp_adjustment": float(
            np.max(np.abs(sim["la_tlmp"] - sim["la_lambda"][None, :]))
        ),
        "rp_bind_share": float(np.mean(sim["rp_bind_up"] | sim["rp_bind_down"])),
        "la_bind_share": float(np.mean(sim["la_bind_up"] | sim["la_bind_down"])),
    }


def generator_frame(sim: dict[str, Any]) -> dict[str, np.ndarray]:
    """Return generator-level response variables and explanatory features."""
    fleet: Fleet = sim["fleet"]
    rp_bind = np.mean(sim["rp_bind_up"] | sim["rp_bind_down"], axis=1)
    la_bind = np.mean(sim["la_bind_up"] | sim["la_bind_down"], axis=1)
    flex = 0.5 * (fleet.ramp_up + fleet.ramp_down) / fleet.pmax
    rp_utilization = np.mean(sim["rp_p"] / fleet.pmax[:, None], axis=1)
    la_utilization = np.mean(sim["la_p"] / fleet.pmax[:, None], axis=1)
    rp_headroom = np.mean((fleet.pmax[:, None] - sim["rp_p"]) / fleet.pmax[:, None], axis=1)
    la_headroom = np.mean((fleet.pmax[:, None] - sim["la_p"]) / fleet.pmax[:, None], axis=1)
    rp_margin_share = np.mean(
        sim["rp_lambda"][None, :] > fleet.cost[:, None] + 1e-7, axis=1
    )
    la_margin_share = np.mean(
        sim["la_lambda"][None, :] > fleet.cost[:, None] + 1e-7, axis=1
    )
    return {
        "generator": np.asarray(fleet.names),
        "cost": fleet.cost.copy(),
        "capacity": fleet.pmax.copy(),
        "flexibility": flex,
        "rp_bind": rp_bind,
        "la_bind": la_bind,
        "rp_scarcity_flex": rp_bind / np.maximum(flex, 1e-9),
        "la_scarcity_flex": la_bind / np.maximum(flex, 1e-9),
        "rp_utilization": rp_utilization,
        "la_utilization": la_utilization,
        "rp_headroom": rp_headroom,
        "la_headroom": la_headroom,
        "rp_margin_share": rp_margin_share,
        "la_margin_share": la_margin_share,
        "loc_rp": sim["rp_loc"].copy(),
        "loc_rp_energy": sim["rp_energy_loc"].copy(),
        "loc_la": sim["la_loc"].copy(),
        "loc_tlmp": sim["tlmp_loc"].copy(),
        "delta_la_rp": sim["la_loc"] - sim["rp_loc"],
    }
