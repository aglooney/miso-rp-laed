import argparse
import copy
import json
import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")  # must be set before importing matplotlib anywhere
import numpy as np
from pyomo.environ import AbstractModel, Constraint, DataPortal, NonNegativeIntegers, NonNegativeReals
from pyomo.environ import Objective, Param, RangeSet, SolverFactory, Var, minimize, value
import pyomo.environ as pyo

import laed_rp_analysis as lra
from laed_rp_analysis import ED_no_errors, LAED_No_Errors, TLMP_calculation, laed_opt_model
from pmp_abstract import build_pmp_abstract_model
from cmp_abstract import build_cmp_abstract_model


def _unit_variance_innovation(error_type: str, rng: np.random.Generator, **kwargs) -> float:
    et = error_type.lower()
    if et in ("gaussian", "normal"):
        return float(rng.standard_normal())
    if et == "laplace":
        # Laplace(0, b) has Var = 2 b^2. Set b=1/sqrt(2) => Var=1.
        return float(rng.laplace(0.0, 1.0 / np.sqrt(2.0)))
    if et in ("student-t", "t"):
        df = float(kwargs.get("df", 3.0))
        if df <= 2:
            raise ValueError("student-t requires df > 2 for finite variance")
        z = rng.standard_t(df)
        # Var(z) = df/(df-2); scale to unit variance.
        return float(z / np.sqrt(df / (df - 2.0)))
    raise ValueError(f"Unknown error_type: {error_type}. Use 'gaussian', 'laplace' or 'student-t'.")


def correlated_error_update(prev_err, std_h, rho, rng, error_type="gaussian", **error_kwargs):
    z = _unit_variance_innovation(error_type=error_type, rng=rng, **error_kwargs)
    return float(rho * prev_err + np.sqrt(max(0.0, 1.0 - rho * rho)) * std_h * z)


def build_forecast_load_window(
    load_true,
    err_state,
    t0,
    horizon,
    sigma_rel,
    rho,
    rng,
    error_type="gaussian",
    error_kwargs=None,
    clip_nonnegative=True,
):
    """
    Exogenous rolling forecast model.

    - At lead h=0 (current interval), forecast error is 0 (perfectly observed).
    - At lead h>0, Std ∝ sqrt(h) and forecasts are correlated revisions across windows.

    Returns: dict {1..horizon} of forecasted load values for the window starting at physical time t0.
    Updates err_state in-place for physical times t0..t0+horizon-1.
    """
    if error_kwargs is None:
        error_kwargs = {}
    load_window = {}

    # Observed at the current time step.
    err_state[t0] = 0.0

    for tt in range(1, horizon + 1):
        t_phys = t0 + tt - 1
        h = tt - 1
        if h == 0:
            e = 0.0
            err_state[t_phys] = 0.0
        else:
            std_h = float(sigma_rel * abs(load_true[t_phys]) * np.sqrt(h))
            e = correlated_error_update(
                prev_err=err_state[t_phys],
                std_h=std_h,
                rho=rho,
                rng=rng,
                error_type=error_type,
                **error_kwargs,
            )
            err_state[t_phys] = e

        Lhat = float(load_true[t_phys]) + float(e)
        if clip_nonnegative:
            Lhat = max(0.0, Lhat)
        load_window[tt] = float(Lhat)

    return load_window


def initialize_gen_init(data, solver):
    """Initialize Gen_init exactly as in laed_rp_analysis main flow."""
    n_g = data.data()["N_g"][None]
    load_ini = data.data()["Load"][1]

    if n_g == 2:
        gen1_ini = min(data.data()["Capacity"][1], load_ini)
        data.data()["Gen_init"] = {
            1: gen1_ini,
            2: min(data.data()["Capacity"][2], load_ini - gen1_ini),
        }
        return

    model_ini = AbstractModel()
    model_ini.N_g = Param(within=NonNegativeIntegers)
    model_ini.G = RangeSet(1, model_ini.N_g)
    model_ini.Cost = Param(model_ini.G)
    model_ini.Capacity = Param(model_ini.G)
    model_ini.reserve_single = Param()
    model_ini.P = Var(model_ini.G, within=NonNegativeReals)
    model_ini.Reserve = Var(model_ini.G, within=NonNegativeReals)

    def objective_rule(model):
        return sum(model.Cost[g] * model.P[g] for g in model.G)

    model_ini.obj = Objective(rule=objective_rule, sense=minimize)

    def power_balance_rule(model):
        return sum(model.P[g] for g in model.G) == load_ini

    model_ini.power_balance_constraint = Constraint(rule=power_balance_rule)

    def capacity_rule(model, g):
        return model.P[g] + model.Reserve[g] <= model.Capacity[g]

    model_ini.capacity_constraint = Constraint(model_ini.G, rule=capacity_rule)

    def reserve_rule(model):
        return sum(model.Reserve[g] for g in model.G) >= lra.reserve_factor * load_ini

    model_ini.reserve_constraint = Constraint(rule=reserve_rule)

    def reserve_single_rule(model, g):
        return model.Reserve[g] <= model.reserve_single * model.Capacity[g]

    model_ini.reserve_single_constraint = Constraint(model_ini.G, rule=reserve_single_rule)

    ed_ini = model_ini.create_instance(data)
    solver.solve(ed_ini, tee=False)
    data.data()["Gen_init"] = {g: ed_ini.P[g].value for g in ed_ini.G}


def apply_aug_2032_projection(data, projection_path, ref_cap):
    with open(projection_path, "r") as f:
        interpolated_data = json.load(f)

    aug_2032_ori = interpolated_data["2032_Aug"]
    scale = ref_cap / (sum(aug_2032_ori.values()) / len(aug_2032_ori))
    aug_2032 = {int(key): value * scale for key, value in aug_2032_ori.items()}
    data.data()["Load"] = aug_2032
    data.data()["N_T"][None] = len(aug_2032)


def series_stats(series):
    delta = np.diff(series)
    return {
        "mean": float(np.mean(series)),
        "std": float(np.std(series)),
        "range": float(np.max(series) - np.min(series)),
        "std_delta": float(np.std(delta)) if delta.size else 0.0,
        "mad_delta": float(np.mean(np.abs(delta))) if delta.size else 0.0,
        "p95_abs_delta": float(np.percentile(np.abs(delta), 95)) if delta.size else 0.0,
    }


def build_scaled_inputs(data, load_factor, ramp_factor):
    base = data.data()
    reserve_single = base["reserve_single"] if isinstance(base["reserve_single"], (int, float)) else base["reserve_single"][None]
    ramp_single = base["ramp_single"] if isinstance(base["ramp_single"], (int, float)) else base["ramp_single"][None]
    return {
        "N_g": base["N_g"][None],
        "N_T": base["N_T"][None],
        "Cost": dict(base["Cost"]),
        "Capacity": dict(base["Capacity"]),
        "Ramp_lim": {g: base["Ramp_lim"][g] * ramp_factor for g in base["Ramp_lim"]},
        "Load": {t: base["Load"][t] * load_factor for t in base["Load"]},
        "Gen_init": {g: base["Gen_init"][g] * load_factor for g in base["Gen_init"]},
        "reserve_single": reserve_single,
        "ramp_single": ramp_single,
    }


def solve_laed_window_with_duals(scaled, n_t, t0, gen_init_current, solver, load_window=None):
    model = laed_opt_model()
    if load_window is None:
        load_window = {tt: scaled["Load"][t0 + tt - 1] for tt in range(1, n_t + 1)}
    inst_data = {
        None: {
            "N_g": {None: scaled["N_g"]},
            "N_t": {None: n_t},
            "N_T": {None: scaled["N_T"]},
            "Cost": scaled["Cost"],
            "Capacity": scaled["Capacity"],
            "Ramp_lim": scaled["Ramp_lim"],
            "Load": load_window,
            "Gen_init": gen_init_current,
            "reserve_single": {None: scaled["reserve_single"]},
        }
    }
    inst = model.create_instance(data=inst_data)
    res = solver.solve(inst, tee=False, load_solutions=False)
    term = str(res.solver.termination_condition).lower()
    status = str(res.solver.status).lower()
    if term != "optimal" and status != "ok":
        raise RuntimeError(f"LAED infeasible/unknown at t0={t0}: status={status}, term={term}")
    inst.solutions.load_from(res)

    p_laed, shed_t, tlmp_t, llmp_t, mu_down, mu_up, _, _ = TLMP_calculation(inst, scaled["N_g"], n_t)
    dispatch_current = {g: float(p_laed[g - 1, 0]) for g in range(1, scaled["N_g"] + 1)}
    mu_up_boundary = np.array([float(mu_up[g - 1, 0]) for g in range(1, scaled["N_g"] + 1)], dtype=float)
    mu_down_boundary = np.array([float(mu_down[g - 1, 0]) for g in range(1, scaled["N_g"] + 1)], dtype=float)
    tlmp_commit = np.array([float(tlmp_t[g - 1, 0]) for g in range(1, scaled["N_g"] + 1)], dtype=float)
    shed_commit = float(shed_t[0]) if np.size(shed_t) else 0.0
    return float(llmp_t[0]), tlmp_commit, dispatch_current, mu_up_boundary, mu_down_boundary, shed_commit


def solve_rp_window_price(scaled, rp_n_t, t0, gen_prev, load_window, solver):
    model = lra.rped_opt_model()
    inst_data = {
        None: {
            "N_g": {None: scaled["N_g"]},
            "N_t": {None: rp_n_t},
            "Cost": scaled["Cost"],
            "Capacity": scaled["Capacity"],
            "Ramp_lim": scaled["Ramp_lim"],
            "Gen_prev": gen_prev,
            "Load": load_window,
            "ramp_single": {None: scaled["ramp_single"]},
            "reserve_single": {None: scaled["reserve_single"]},
        }
    }
    inst = model.create_instance(data=inst_data)
    res = solver.solve(inst, tee=False, load_solutions=False)
    term = str(res.solver.termination_condition).lower()
    status = str(res.solver.status).lower()
    if term != "optimal" and status != "ok":
        raise RuntimeError(f"RP infeasible/unknown at t0={t0}: status={status}, term={term}")
    inst.solutions.load_from(res)

    p_val, shed_commit, lmp, *_rest = lra.LMP_calculation(inst)
    dispatch_current = {g: float(p_val[g - 1]) for g in range(1, scaled["N_g"] + 1)}
    return float(lmp), dispatch_current, float(shed_commit)


def solve_prices_with_forecast_error(
    scaled,
    laed_n_t,
    rp_n_t,
    n_steps,
    solver,
    sigma_rel,
    rho,
    seed,
    error_type="gaussian",
    error_kwargs=None,
):
    """
    Return price series under a common exogenous forecast process:
      - RP LMP (system)
      - LAED LMP (system)
      - LAED TLMP (per generator)
    """
    if error_kwargs is None:
        error_kwargs = {}

    rng = np.random.default_rng(int(seed))
    n_g = int(scaled["N_g"])
    rp_lmp = np.zeros(n_steps, dtype=float)
    laed_lmp = np.zeros(n_steps, dtype=float)
    laed_tlmp = np.zeros((n_g, n_steps), dtype=float)
    pmp_price = np.zeros(n_steps, dtype=float)
    cmp_price = np.zeros(n_steps, dtype=float)
    pmp_shed = np.zeros(n_steps, dtype=float)
    cmp_shed = np.zeros(n_steps, dtype=float)
    rp_shed = np.zeros(n_steps, dtype=float)
    laed_shed = np.zeros(n_steps, dtype=float)
    ramp_bind_count = 0

    gen_prev_rp = dict(scaled["Gen_init"])
    gen_prev_laed = dict(scaled["Gen_init"])
    # PMP needs an additional lag because its internal horizon includes one relaxed past interval.
    gen_prev_pmp = dict(scaled["Gen_init"])      # dispatch at physical t0-1 (rolling memory)
    gen_prev2_pmp = dict(scaled["Gen_init"])     # dispatch at physical t0-2 (rolling memory)
    lambda_prev = 0.0
    gen_prev_cmp = dict(scaled["Gen_init"])

    # Persistent error state per physical time index (1..N_T)
    err_state = {t: 0.0 for t in range(1, int(scaled["N_T"]) + 1)}

    for k in range(n_steps):
        t0 = k + 1
        load_full = build_forecast_load_window(
            load_true=scaled["Load"],
            err_state=err_state,
            t0=t0,
            horizon=laed_n_t,
            sigma_rel=sigma_rel,
            rho=rho,
            rng=rng,
            error_type=error_type,
            error_kwargs=error_kwargs,
            clip_nonnegative=True,
        )

        load_rp = {tt: load_full[tt] for tt in range(1, rp_n_t + 1)}
        rp_lmp_k, dispatch_rp, shed_rp_k = solve_rp_window_price(
            scaled, rp_n_t, t0, gen_prev_rp, load_rp, solver
        )
        rp_lmp[k] = rp_lmp_k
        rp_shed[k] = shed_rp_k
        gen_prev_rp = dispatch_rp

        llmp_k, tlmp_vec, dispatch_laed, mu_up_b, mu_down_b, shed_laed_k = solve_laed_window_with_duals(
            scaled, laed_n_t, t0, gen_prev_laed, solver, load_window=load_full
        )
        laed_lmp[k] = llmp_k
        laed_tlmp[:, k] = tlmp_vec
        laed_shed[k] = shed_laed_k
        if np.any(np.array(mu_up_b) > 1e-8) or np.any(np.array(mu_down_b) > 1e-8):
            ramp_bind_count += 1
        gen_prev_laed = dispatch_laed

        # PMP on the same exogenous forecast:
        # - lag load is realized (no error) because it's already committed by the time we solve window t0
        load_lag = float(scaled["Load"][t0 - 1] if t0 > 1 else load_full[1])
        pmp_k, dispatch_pmp, shed_pmp_k = solve_pmp_window_forecast(
            scaled=scaled,
            n_t=laed_n_t,
            t0=t0,
            gen_init=gen_prev2_pmp,
            lambda_prev=lambda_prev,
            load_window=load_full,
            load_lag=load_lag,
            solver=solver,
        )
        pmp_price[k] = float(pmp_k)
        pmp_shed[k] = float(shed_pmp_k)
        # shift lag memories forward
        gen_prev2_pmp = gen_prev_pmp
        gen_prev_pmp = dispatch_pmp
        lambda_prev = float(pmp_k)

        # CMP on the same exogenous forecast, using LAED boundary ramp duals.
        cmp_k, dispatch_cmp, shed_cmp_k = solve_cmp_window_forecast(
            scaled=scaled,
            n_t=laed_n_t,
            t0=t0,
            p_past_star=gen_prev_cmp,
            mu_ru_star=mu_up_b,
            mu_rd_star=mu_down_b,
            load_window=load_full,
            solver=solver,
        )
        cmp_price[k] = float(cmp_k)
        cmp_shed[k] = float(shed_cmp_k)
        gen_prev_cmp = dispatch_cmp

    diagnostics = {
        "rp_shed": rp_shed,
        "laed_shed": laed_shed,
        "pmp_shed": pmp_shed,
        "cmp_shed": cmp_shed,
        "ramp_bind_count": ramp_bind_count,
    }
    return rp_lmp, laed_lmp, laed_tlmp, pmp_price, cmp_price, diagnostics


def solve_pmp_window(scaled, n_t, t0, gen_prev, lambda_prev, solver):
    """
    Rolling PMP implementation:
    - includes one lag interval (h=1) with Lagrangian settlement at lambda_prev
    - current physical interval is h=2; price is dual on balance[h=2]
    """
    n_g = scaled["N_g"]
    n_t_eff = n_t + 1

    # Map PMP time indices:
    #  t=1 is the lag interval (physical t0-1), relaxed via Lambda_star
    #  t=2 is the committed/current interval (physical t0), enforced by Balance[2]
    lag_load = scaled["Load"][t0 - 1] if t0 > 1 else scaled["Load"][t0]

    load_map = {1: float(lag_load)}
    for tt in range(2, n_t_eff + 1):
        load_map[tt] = float(scaled["Load"][t0 + (tt - 2)])

    lambda_star = {t: 0.0 for t in range(1, n_t_eff + 1)}
    lambda_star[1] = float(lambda_prev)

    inst_data = {
        None: {
            "N_g": {None: n_g},
            "N_T": {None: n_t_eff},
            "T0": {None: 1},
            "t_hat": {None: 2},
            "T_end": {None: n_t_eff},
            "Cost": scaled["Cost"],
            "Capacity": scaled["Capacity"],
            "Ramp_lim": scaled["Ramp_lim"],
            "Load": load_map,
            "Gen_init": gen_prev,
            "Lambda_star": lambda_star,
            "cost_load": {None: float(lra.cost_load)},
        }
    }

    model = build_pmp_abstract_model(allow_shedding=True)
    m = model.create_instance(data=inst_data)

    res = solver.solve(m, tee=False, load_solutions=False)
    term = str(res.solver.termination_condition).lower()
    status = str(res.solver.status).lower()
    if term != "optimal" and status != "ok":
        raise RuntimeError(f"PMP infeasible/unknown at t0={t0}: status={status}, term={term}")
    m.solutions.load_from(res)

    price_current = float(m.dual.get(m.Balance[2], 0.0))
    dispatch_current = {g: float(value(m.P[g, 2])) for g in m.G}
    shed_current = float(value(m.Loadshed[2])) if hasattr(m, "Loadshed") else 0.0
    return price_current, dispatch_current, shed_current


def solve_pmp_window_forecast(scaled, n_t, t0, gen_init, lambda_prev, load_window, load_lag, solver):
    """
    PMP under an exogenous forecast:
      - t=1 is lag (physical t0-1), relaxed in objective using Lambda_star[1]=lambda_prev
      - t=2 is current/committed (physical t0), price is dual(Balance[2])
      - t>=3 are future intervals (physical t0+1, ...), balance enforced

    load_window: dict {1..n_t} forecast loads for physical t0..t0+n_t-1
    load_lag: scalar realized load for physical t0-1
    """
    n_g = int(scaled["N_g"])
    n_t_eff = int(n_t) + 1

    load_map = {1: float(load_lag)}
    for tt in range(2, n_t_eff + 1):
        load_map[tt] = float(load_window[tt - 1])

    lambda_star = {t: 0.0 for t in range(1, n_t_eff + 1)}
    lambda_star[1] = float(lambda_prev)

    inst_data = {
        None: {
            "N_g": {None: n_g},
            "N_T": {None: n_t_eff},
            "T0": {None: 1},
            "t_hat": {None: 2},
            "T_end": {None: n_t_eff},
            "Cost": scaled["Cost"],
            "Capacity": scaled["Capacity"],
            "Ramp_lim": scaled["Ramp_lim"],
            "Load": load_map,
            "Gen_init": gen_init,
            "Lambda_star": lambda_star,
            "cost_load": {None: float(lra.cost_load)},
        }
    }

    model = build_pmp_abstract_model(allow_shedding=True)
    m = model.create_instance(data=inst_data)

    res = solver.solve(m, tee=False, load_solutions=False)
    term = str(res.solver.termination_condition).lower()
    status = str(res.solver.status).lower()
    if term != "optimal" and status != "ok":
        raise RuntimeError(f"PMP infeasible/unknown at t0={t0}: status={status}, term={term}")
    m.solutions.load_from(res)

    price_current = float(m.dual.get(m.Balance[2], 0.0))
    dispatch_current = {g: float(value(m.P[g, 2])) for g in m.G}
    shed_current = float(value(m.Loadshed[2])) if hasattr(m, "Loadshed") else 0.0
    return price_current, dispatch_current, shed_current


def solve_cmp_window_forecast(
    scaled,
    n_t,
    t0,
    p_past_star,
    mu_ru_star,
    mu_rd_star,
    load_window,
    solver,
):
    """
    CMP under an exogenous forecast.

    This uses cmp_abstract.py:
      - Future horizon only, indexed t=1..n_t
      - Coupling to past dispatch via P_past_star in ramp constraints at t_hat=1
      - Objective is operational cost plus the LAED boundary ramp-dual adjustment term.

    Returns: (price_current, dispatch_current)
      - price_current is dual(Balance[1]) (i.e., at the committed interval)
    """
    n_g = int(scaled["N_g"])
    n_t = int(n_t)

    inst_data = {
        None: {
            "N_g": {None: n_g},
            "N_T": {None: n_t},
            "t_hat": {None: 1},
            "T_end": {None: n_t},
            "Cost": scaled["Cost"],
            "Capacity": scaled["Capacity"],
            "Ramp_lim": scaled["Ramp_lim"],
            "Load": {tt: float(load_window[tt]) for tt in range(1, n_t + 1)},
            "P_past_star": {g: float(p_past_star[g]) for g in range(1, n_g + 1)},
            "mu_ru_star": {g: float(mu_ru_star[g - 1]) for g in range(1, n_g + 1)},
            "mu_rd_star": {g: float(mu_rd_star[g - 1]) for g in range(1, n_g + 1)},
            "cost_load": {None: float(lra.cost_load)},
        }
    }

    model = build_cmp_abstract_model(allow_shedding=True)
    m = model.create_instance(data=inst_data)

    res = solver.solve(m, tee=False, load_solutions=False)
    term = str(res.solver.termination_condition).lower()
    status = str(res.solver.status).lower()
    if term != "optimal" and status != "ok":
        raise RuntimeError(f"CMP infeasible/unknown at t0={t0}: status={status}, term={term}")
    m.solutions.load_from(res)

    price_current = float(m.dual.get(m.Balance[1], 0.0))
    dispatch_current = {g: float(value(m.P[g, 1])) for g in m.G}
    shed_current = float(value(m.Loadshed[1])) if hasattr(m, "Loadshed") else 0.0
    return price_current, dispatch_current, shed_current


def solve_cmp_window(scaled, n_t, t0, gen_prev, mu_up_boundary, mu_down_boundary, solver):
    """
    Rolling CMP implementation:
    - enforces full future balance/capacity/ramping
    - relaxes only boundary ramp with nonnegative slacks
    - penalizes boundary slacks by LAED boundary ramp dual magnitudes
    """
    n_g = scaled["N_g"]
    m = pyo.ConcreteModel()
    m.G = pyo.RangeSet(1, n_g)
    m.H = pyo.RangeSet(1, n_t)

    load_map = {h: scaled["Load"][t0 + h - 1] for h in m.H}
    m.Cost = pyo.Param(m.G, initialize=scaled["Cost"])
    m.Capacity = pyo.Param(m.G, initialize=scaled["Capacity"])
    m.Ramp_lim = pyo.Param(m.G, initialize=scaled["Ramp_lim"])
    m.Load = pyo.Param(m.H, initialize=load_map)

    m.P = pyo.Var(m.G, m.H, within=pyo.NonNegativeReals)
    m.Loadshed = pyo.Var(m.H, within=pyo.NonNegativeReals)
    m.s_up = pyo.Var(m.G, within=pyo.NonNegativeReals)
    m.s_dn = pyo.Var(m.G, within=pyo.NonNegativeReals)
    m.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT_EXPORT)

    def obj_rule(model):
        gen_cost = sum(model.Cost[g] * model.P[g, h] for g in model.G for h in model.H)
        shed_cost = lra.cost_load * sum(model.Loadshed[h] for h in model.H)
        boundary_pen = sum(abs(mu_up_boundary[g - 1]) * model.s_up[g] for g in model.G) + \
            sum(abs(mu_down_boundary[g - 1]) * model.s_dn[g] for g in model.G)
        return gen_cost + shed_cost + boundary_pen

    m.obj = pyo.Objective(rule=obj_rule, sense=pyo.minimize)

    def bal_rule(model, h):
        return sum(model.P[g, h] for g in model.G) == model.Load[h] - model.Loadshed[h]

    m.balance = pyo.Constraint(m.H, rule=bal_rule)

    def cap_rule(model, g, h):
        return model.P[g, h] <= model.Capacity[g]

    m.capacity = pyo.Constraint(m.G, m.H, rule=cap_rule)

    def ramp_up_boundary(model, g):
        return model.P[g, 1] - gen_prev[g] <= model.Ramp_lim[g] + model.s_up[g]

    def ramp_down_boundary(model, g):
        return gen_prev[g] - model.P[g, 1] <= model.Ramp_lim[g] + model.s_dn[g]

    m.ramp_up_boundary = pyo.Constraint(m.G, rule=ramp_up_boundary)
    m.ramp_down_boundary = pyo.Constraint(m.G, rule=ramp_down_boundary)

    def ramp_up_future(model, g, h):
        if h == 1:
            return pyo.Constraint.Skip
        return model.P[g, h] - model.P[g, h - 1] <= model.Ramp_lim[g]

    def ramp_down_future(model, g, h):
        if h == 1:
            return pyo.Constraint.Skip
        return model.P[g, h - 1] - model.P[g, h] <= model.Ramp_lim[g]

    m.ramp_up_future = pyo.Constraint(m.G, m.H, rule=ramp_up_future)
    m.ramp_down_future = pyo.Constraint(m.G, m.H, rule=ramp_down_future)

    res = solver.solve(m, tee=False, load_solutions=False)
    term = str(res.solver.termination_condition).lower()
    status = str(res.solver.status).lower()
    if term != "optimal" and status != "ok":
        raise RuntimeError(f"CMP infeasible/unknown at t0={t0}: status={status}, term={term}")
    m.solutions.load_from(res)

    price_current = float(m.dual.get(m.balance[1], 0.0))
    dispatch_current = {g: float(value(m.P[g, 1])) for g in m.G}
    s_up_vals = np.array([float(value(m.s_up[g])) for g in m.G], dtype=float)
    s_dn_vals = np.array([float(value(m.s_dn[g])) for g in m.G], dtype=float)
    return price_current, dispatch_current, s_up_vals, s_dn_vals


def solve_rolling_pmp_cmp(scaled, n_t, n_steps, solver):
    pmp = np.zeros(n_steps, dtype=float)
    cmp = np.zeros(n_steps, dtype=float)
    mu_up_norm = np.zeros(n_steps, dtype=float)
    mu_down_norm = np.zeros(n_steps, dtype=float)
    cmp_slack_up = np.zeros(n_steps, dtype=float)
    cmp_slack_down = np.zeros(n_steps, dtype=float)

    gen_prev_pmp = dict(scaled["Gen_init"])
    gen_prev_cmp = dict(scaled["Gen_init"])
    lambda_prev = 0.0

    for k in range(n_steps):
        t0 = k + 1

        llmp_k, _tlmp_vec, dispatch_laed, mu_up_k, mu_down_k, _shed_commit = solve_laed_window_with_duals(
            scaled, n_t, t0, gen_prev_cmp, solver
        )
        _ = llmp_k

        pmp_k, dispatch_pmp, _shed_pmp_k = solve_pmp_window(scaled, n_t, t0, gen_prev_pmp, lambda_prev, solver)
        cmp_k, dispatch_cmp, s_up_vals, s_dn_vals = solve_cmp_window(
            scaled, n_t, t0, gen_prev_cmp, mu_up_k, mu_down_k, solver
        )

        pmp[k] = pmp_k
        cmp[k] = cmp_k
        mu_up_norm[k] = float(np.linalg.norm(mu_up_k, ord=1))
        mu_down_norm[k] = float(np.linalg.norm(mu_down_k, ord=1))
        cmp_slack_up[k] = float(np.sum(s_up_vals))
        cmp_slack_down[k] = float(np.sum(s_dn_vals))

        gen_prev_pmp = dispatch_pmp
        gen_prev_cmp = dispatch_cmp
        lambda_prev = pmp_k

    diagnostics = {
        "mu_up_norm": mu_up_norm,
        "mu_down_norm": mu_down_norm,
        "cmp_slack_up": cmp_slack_up,
        "cmp_slack_down": cmp_slack_down,
    }
    return pmp, cmp, diagnostics


def main():
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser(description="Compare RP/LAED/PMP/CMP price volatility.")
    parser.add_argument("--case", default="toy_data.dat", help="Path to .dat case file.")
    parser.add_argument("--projection", default="MISO_Projection.json", help="Path to load projection JSON.")
    parser.add_argument("--ref-cap", type=float, default=350.0, help="Mean demand target after scaling.")
    parser.add_argument("--n-t", type=int, default=13, help="Rolling look-ahead window length.")
    parser.add_argument("--rp-n-t", type=int, default=2, help="RP horizon length.")
    parser.add_argument("--ramp-factor", type=float, default=0.1806, help="Ramp limit multiplier.")
    parser.add_argument("--load-factor", type=float, default=1.0, help="Load multiplier.")
    parser.add_argument("--cost-load", type=float, default=1e10, help="Load shedding penalty.")
    parser.add_argument("--sigma", type=float, default=0.0, help="Relative 1-step forecast error scale.")
    parser.add_argument("--rho", type=float, default=0.9, help="Correlation of forecast revisions across windows.")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed for forecast errors.")
    parser.add_argument("--error-type", default="gaussian", choices=["gaussian", "laplace", "student-t"])
    parser.add_argument("--df", type=float, default=3.0, help="Student-t dof (only used if --error-type=student-t).")
    parser.add_argument("--output", default="price_volatility.png", help="Output figure path.")
    args = parser.parse_args()

    solver = SolverFactory("gurobi_direct")
    solver.options["OutputFlag"] = 0
    lra.reserve_factor = 0
    lra.cost_load = args.cost_load

    data = DataPortal()
    data.load(filename=args.case)
    apply_aug_2032_projection(data, args.projection, args.ref_cap)
    data.data()["N_t"][None] = args.n_t
    initialize_gen_init(data, solver)

    n_g = data.data()["N_g"][None]
    n_t = args.n_t
    n_t_total = data.data()["N_T"][None]
    n_steps = n_t_total - n_t + 1

    data_rp = copy.deepcopy(data)
    data_laed = copy.deepcopy(data)

    print("Solving RP/LAED/PMP/CMP with forecast error...")
    scaled = build_scaled_inputs(data, args.load_factor, args.ramp_factor)
    err_kwargs = {"df": args.df} if args.error_type == "student-t" else {}
    rp_lmp, laed_lmp, laed_tlmp, pmp_price, cmp_price, diag = solve_prices_with_forecast_error(
        scaled=scaled,
        laed_n_t=n_t,
        rp_n_t=args.rp_n_t,
        n_steps=n_steps,
        solver=solver,
        sigma_rel=args.sigma,
        rho=args.rho,
        seed=args.seed,
        error_type=args.error_type,
        error_kwargs=err_kwargs,
    )

    max_rp_shed = float(np.max(diag["rp_shed"]))
    max_laed_shed = float(np.max(diag["laed_shed"]))
    max_pmp_shed = float(np.max(diag.get("pmp_shed", np.array([0.0]))))
    max_cmp_shed = float(np.max(diag.get("cmp_shed", np.array([0.0]))))
    if max_rp_shed > 1e-6 or max_laed_shed > 1e-6 or max_pmp_shed > 1e-6 or max_cmp_shed > 1e-6:
        print(
            "WARNING: Load shedding detected. Prices may be dominated by the load-shed penalty.\n"
            f"  max(RP shed)={max_rp_shed:.6f} MW\n"
            f"  max(LAED shed)={max_laed_shed:.6f} MW\n"
            f"  max(PMP shed)={max_pmp_shed:.6f} MW\n"
            f"  max(CMP shed)={max_cmp_shed:.6f} MW"
        )
    if diag["ramp_bind_count"] == 0:
        print(
            "WARNING: No binding ramp constraints detected in LAED boundary intervals. "
            "TLMP will likely match LMP. Decrease --ramp-factor (or increase --ref-cap) to induce ramp scarcity."
        )

    x = np.arange(1, n_steps + 1)

    # Quick sanity check: if we detected binding ramps, TLMP should differ from LMP at least sometimes.
    if diag["ramp_bind_count"] > 0 and not np.any(np.abs(laed_tlmp - laed_lmp[None, :]) > 1e-8):
        print(
            "WARNING: LAED TLMP is identical to LAED LMP for both generators despite ramp binds being detected. "
            "Double-check dual sign conventions and the TLMP formula."
        )

    lmp_diff = rp_lmp - laed_lmp
    print(
        "LMP diff (RP - LAED): max_abs, nonzero_count\n"
        f"{np.max(np.abs(lmp_diff)):.6f}, {int(np.sum(np.abs(lmp_diff) > 1e-9))}"
    )

    plt.figure(figsize=(12, 5))
    plt.plot(x, rp_lmp, label="RP LMP", alpha=0.9, linestyle="--", linewidth=1.5)
    plt.plot(x, laed_lmp, label="LAED LMP", alpha=0.9, linestyle="-", linewidth=1.5)
    plt.plot(x, pmp_price, label="PMP Price", alpha=0.9, linestyle=":", linewidth=1.8)
    plt.plot(x, cmp_price, label="CMP Price", alpha=0.9, linestyle="-.", linewidth=1.6)
    plt.plot(x, laed_tlmp[0, :], label="LAED TLMP Gen 1", alpha=0.9)
    #plt.plot(x, laed_tlmp[1, :], label="LAED TLMP Gen 2", alpha=0.9)
    plt.xlabel("Time (5-min Intervals)")
    plt.ylabel("Price ($/MWh)")
    plt.title("Price Trajectories with Forecast Error")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.output, dpi=200)

    rows = [
        ("RP_LMP", series_stats(rp_lmp)),
        ("LAED_LMP", series_stats(laed_lmp)),
        ("PMP", series_stats(pmp_price)),
        ("CMP", series_stats(cmp_price)),
        ("LAED_TLMP_G1", series_stats(laed_tlmp[0, :])),
        ("LAED_TLMP_G2", series_stats(laed_tlmp[1, :])),
    ]
    print("name, mean, std, range, std_delta, mad_delta, p95_abs_delta")
    for name, stats in rows:
        print(
            f"{name}, {stats['mean']:.6f}, {stats['std']:.6f}, {stats['range']:.6f}, "
            f"{stats['std_delta']:.6f}, {stats['mad_delta']:.6f}, {stats['p95_abs_delta']:.6f}"
        )

    # Volatility summary plot (bars). This is often easier to compare than overlaying many time series.
    # We intentionally report both level volatility (std) and change volatility (std of deltas).
    labels = [name.replace("_", " ") for name, _ in rows]
    std_level = [stats["std"] for _name, stats in rows]
    std_delta = [stats["std_delta"] for _name, stats in rows]
    mad_delta = [stats["mad_delta"] for _name, stats in rows]
    p95_delta = [stats["p95_abs_delta"] for _name, stats in rows]

    out_base, out_ext = os.path.splitext(args.output)
    metrics_path = out_base + "_volatility_metrics" + (out_ext or ".png")

    fig, axs = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
    axs = axs.ravel()
    x_idx = np.arange(len(labels), dtype=int)

    def _bar(ax, y, title, ylabel):
        ax.bar(x_idx, y, color="#4C78A8", alpha=0.9)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(True, axis="y", alpha=0.25)

    _bar(axs[0], std_level, "Std Dev of Price Level", "Std(price) ($/MWh)")
    _bar(axs[1], std_delta, "Std Dev of Price Changes", "Std(Δprice) ($/MWh per interval)")
    _bar(axs[2], mad_delta, "Mean Absolute Price Change", "Mean|Δprice| ($/MWh per interval)")
    _bar(axs[3], p95_delta, "95th Percentile Absolute Price Change", "P95|Δprice| ($/MWh per interval)")

    for ax in axs[2:]:
        ax.set_xticks(x_idx)
        ax.set_xticklabels(labels, rotation=25, ha="right")

    fig.suptitle("Price Volatility Summary (forecast error)")
    fig.tight_layout()
    fig.savefig(metrics_path, dpi=200)
    print(f"Saved volatility metrics figure to: {metrics_path}")
    print(f"Saved figure to: {args.output}")
    print(
        "Diagnostics: max(RP shed), max(LAED shed), max(PMP shed), max(CMP shed), count(LAED boundary ramp binds)\n"
        f"{np.max(diag['rp_shed']):.6f}, {np.max(diag['laed_shed']):.6f}, "
        f"{np.max(diag['pmp_shed']):.6f}, {np.max(diag['cmp_shed']):.6f}, {diag['ramp_bind_count']}"
    )


if __name__ == "__main__":
    main()
