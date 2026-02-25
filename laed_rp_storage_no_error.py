import pyomo.environ as pyo
from pyomo.environ import (
    AbstractModel, Param, RangeSet, Var, Constraint, Objective, Suffix, minimize, DataPortal,
    NonNegativeIntegers, NonNegativeReals, value, SolverFactory
)
import numpy as np
import copy
import matplotlib.pyplot as plt
import json
from tqdm import tqdm


# -------- Storage helpers (no stochastic errors) --------
def _extract_storage_inputs(base: dict):
    """Return storage parameter dictionaries with safe defaults if storage is absent."""
    n_s = int(base.get("N_s", {}).get(None, 0) if isinstance(base.get("N_s", {}), dict) else base.get("N_s", 0) or 0)
    S_idx = range(1, n_s + 1)

    def _by_s(name, default=0.0):
        if name in base:
            val = base[name]
            return dict(val) if isinstance(val, dict) else {}
        return {s: default for s in S_idx}

    E_cap = _by_s("E_cap", 0.0)
    P_ch_cap = _by_s("P_ch_cap", 0.0)
    P_dis_cap = _by_s("P_dis_cap", 0.0)
    SoC_init = _by_s("SoC_init", 0.0)
    SoC_min = _by_s("SoC_min", 0.0)
    SoC_max = _by_s("SoC_max", 0.0) if "SoC_max" in base else {s: E_cap.get(s, 0.0) for s in S_idx}

    charge_cost = _by_s("charge_cost", 0.0)
    discharge_cost = _by_s("discharge_cost", 0.0)

    eta_c = float(base.get("eta_c", {}).get(None, 1.0) if isinstance(base.get("eta_c", {}), dict) else base.get("eta_c", 1.0))
    eta_d = float(base.get("eta_d", {}).get(None, 1.0) if isinstance(base.get("eta_d", {}), dict) else base.get("eta_d", 1.0))
    delta_t = float(base.get("delta_t", {}).get(None, 1.0) if isinstance(base.get("delta_t", {}), dict) else base.get("delta_t", 1.0))

    return {
        "N_s": n_s,
        "E_cap": E_cap,
        "P_ch_cap": P_ch_cap,
        "P_dis_cap": P_dis_cap,
        "SoC_init": SoC_init,
        "SoC_min": SoC_min,
        "SoC_max": SoC_max,
        "charge_cost": charge_cost,
        "discharge_cost": discharge_cost,
        "eta_c": eta_c,
        "eta_d": eta_d,
        "delta_t": delta_t,
    }


# -------- Models with storage (deterministic) --------
def rped_opt_model():
    m = AbstractModel()
    m.N_g = Param(within=NonNegativeIntegers)
    m.N_t = Param(within=NonNegativeIntegers)
    m.N_s = Param(within=NonNegativeIntegers, default=0, initialize=0)
    m.G = RangeSet(1, m.N_g)
    m.T = RangeSet(1, m.N_t)
    m.S = RangeSet(1, m.N_s)

    m.Cost = Param(m.G)
    m.Capacity = Param(m.G)
    m.Ramp_lim = Param(m.G)
    m.Load = Param(m.T)
    m.Gen_prev = Param(m.G)
    m.ramp_single = Param()
    m.reserve_single = Param()

    m.E_cap = Param(m.S, default=0.0, initialize=lambda m, s: 0.0)
    m.P_ch_cap = Param(m.S, default=0.0, initialize=lambda m, s: 0.0)
    m.P_dis_cap = Param(m.S, default=0.0, initialize=lambda m, s: 0.0)
    m.SoC_init = Param(m.S, default=0.0, initialize=lambda m, s: 0.0)
    m.SoC_min = Param(m.S, default=0.0, initialize=lambda m, s: 0.0)
    m.SoC_max = Param(m.S, default=0.0, initialize=lambda m, s: 0.0)
    m.eta_c = Param(default=1.0, initialize=1.0)
    m.eta_d = Param(default=1.0, initialize=1.0)
    m.charge_cost = Param(m.S, default=0.0, initialize=lambda m, s: 0.0)
    m.discharge_cost = Param(m.S, default=0.0, initialize=lambda m, s: 0.0)
    m.delta_t = Param(default=1.0, initialize=1.0)

    m.P = Var(m.G, within=NonNegativeReals)
    m.Rampup = Var(m.G, m.T, within=NonNegativeReals)
    m.Rampdown = Var(m.G, m.T, within=NonNegativeReals)
    m.Loadshed = Var(m.T, within=NonNegativeReals)
    m.P_charge = Var(m.S, m.T, within=NonNegativeReals)
    m.P_discharge = Var(m.S, m.T, within=NonNegativeReals)
    m.SoC = Var(m.S, m.T, within=NonNegativeReals)

    def objective_rule(m):
        gen_cost = sum(m.Cost[g] * m.P[g] for g in m.G)
        # RP/ED is a single-interval energy dispatch + a ramp product over the look-ahead window.
        # Only the commit interval (t=1) is energy-dispatched. Treat t>1 storage variables
        # as capability endpoints for the ramp product, not paid energy.
        shed_cost = cost_load * m.Loadshed[1]
        storage_cost = sum(
            m.charge_cost[s] * m.P_charge[s, 1] + m.discharge_cost[s] * m.P_discharge[s, 1]
            for s in m.S
        )
        return gen_cost + shed_cost + storage_cost
    m.obj = Objective(rule=objective_rule, sense=minimize)

    # Keep future shedding at 0 so the model can't relax ramp requirements by shedding
    # in non-committed intervals.
    def future_shed_zero_rule(m, t):
        if t == 1:
            return Constraint.Skip
        return m.Loadshed[t] == 0.0
    m.future_shed_zero_constraint = Constraint(m.T, rule=future_shed_zero_rule)

    def capacity_up_rule(m, g, t):
        return m.P[g] + m.Rampup[g, t] <= m.Capacity[g]
    m.capacity_up_constraint = Constraint(m.G, m.T, rule=capacity_up_rule)

    def capacity_down_rule(m, g, t):
        return m.P[g] - m.Rampdown[g, t] >= 0
    m.capacity_down_constraint = Constraint(m.G, m.T, rule=capacity_down_rule)

    def power_balance_rule(m):
        net_storage = sum(m.P_discharge[s, 1] - m.P_charge[s, 1] for s in m.S)
        return sum(m.P[g] for g in m.G) + net_storage == m.Load[1] - m.Loadshed[1]
    m.power_balance_constraint = Constraint(rule=power_balance_rule)

    def ramp_down_rule(m, g):
        return m.P[g] - m.Gen_prev[g] >= - m.Ramp_lim[g]
    m.ramp_down_constraint = Constraint(m.G, rule=ramp_down_rule)

    def ramp_up_rule(m, g):
        return m.P[g] - m.Gen_prev[g] <= m.Ramp_lim[g]
    m.ramp_up_constraint = Constraint(m.G, rule=ramp_up_rule)

    def rampup_window_rule(m, g, t):
        # Only a single endpoint ramp product (at t = last).
        if t == 1:
            return m.Rampup[g, t] <= 0.0
        if t == m.T.last():
            return m.Rampup[g, t] <= (t - 1) * m.ramp_single * m.Ramp_lim[g]
        return m.Rampup[g, t] <= 0.0
    m.rampup_window_constraint = Constraint(m.G, m.T, rule=rampup_window_rule)

    def rampdown_window_rule(m, g, t):
        # Only a single endpoint ramp product (at t = last).
        if t == 1:
            return m.Rampdown[g, t] <= 0.0
        if t == m.T.last():
            return m.Rampdown[g, t] <= (t - 1) * m.ramp_single * m.Ramp_lim[g]
        return m.Rampdown[g, t] <= 0.0
    m.rampdown_window_constraint = Constraint(m.G, m.T, rule=rampdown_window_rule)

    def storage_charge_cap_rule(m, s, t):
        return m.P_charge[s, t] <= m.P_ch_cap[s]
    m.storage_charge_cap_constraint = Constraint(m.S, m.T, rule=storage_charge_cap_rule)

    def storage_discharge_cap_rule(m, s, t):
        return m.P_discharge[s, t] <= m.P_dis_cap[s]
    m.storage_discharge_cap_constraint = Constraint(m.S, m.T, rule=storage_discharge_cap_rule)

    def storage_charge_window_rule(m, s, t):
        # Allow storage operation at:
        # - t=1 (commit/energy balance)
        # - t=last (ramp-product endpoint)
        if t in (1, m.T.last()):
            return Constraint.Skip
        return m.P_charge[s, t] == 0.0
    m.storage_charge_window_constraint = Constraint(m.S, m.T, rule=storage_charge_window_rule)

    def storage_discharge_window_rule(m, s, t):
        # Allow storage operation at:
        # - t=1 (commit/energy balance)
        # - t=last (ramp-product endpoint)
        if t in (1, m.T.last()):
            return Constraint.Skip
        return m.P_discharge[s, t] == 0.0
    m.storage_discharge_window_constraint = Constraint(m.S, m.T, rule=storage_discharge_window_rule)

    def storage_energy_bounds_rule(m, s, t):
        return pyo.inequality(m.SoC_min[s], m.SoC[s, t], m.SoC_max[s])
    m.storage_energy_bounds_constraint = Constraint(m.S, m.T, rule=storage_energy_bounds_rule)

    def storage_soc_balance_rule(m, s, t):
        dt = m.delta_t
        if t == 1:
            return m.SoC[s, t] == m.SoC_init[s] + m.eta_c * m.P_charge[s, t] * dt - (1 / m.eta_d) * m.P_discharge[s, t] * dt
        return m.SoC[s, t] == m.SoC[s, t-1] + m.eta_c * m.P_charge[s, t] * dt - (1 / m.eta_d) * m.P_discharge[s, t] * dt
    m.storage_soc_balance_constraint = Constraint(m.S, m.T, rule=storage_soc_balance_rule)

    def storage_no_simul_rule(m, s, t):
        cap = max(float(pyo.value(m.P_ch_cap[s])), float(pyo.value(m.P_dis_cap[s])))
        return m.P_charge[s, t] + m.P_discharge[s, t] <= cap
    m.storage_no_simul_constraint = Constraint(m.S, m.T, rule=storage_no_simul_rule)

    def ru_endpoint_rule(m):
        t1 = 1
        tN = m.T.last()
        if tN < 2:
            return Constraint.Skip
        storage_net_t1 = sum(m.P_discharge[s, t1] - m.P_charge[s, t1] for s in m.S)
        storage_net_tN = sum(m.P_discharge[s, tN] - m.P_charge[s, tN] for s in m.S)
        return sum(m.Rampup[g, tN] for g in m.G) + (storage_net_tN - storage_net_t1) >= (m.Load[tN] - m.Load[t1])
    m.ru_endpoint_constraint = Constraint(rule=ru_endpoint_rule)

    def rd_endpoint_rule(m):
        t1 = 1
        tN = m.T.last()
        if tN < 2:
            return Constraint.Skip
        storage_net_t1 = sum(m.P_discharge[s, t1] - m.P_charge[s, t1] for s in m.S)
        storage_net_tN = sum(m.P_discharge[s, tN] - m.P_charge[s, tN] for s in m.S)
        return sum(m.Rampdown[g, tN] for g in m.G) - (storage_net_tN - storage_net_t1) >= (m.Load[t1] - m.Load[tN])
    m.rd_endpoint_constraint = Constraint(rule=rd_endpoint_rule)

    m.dual = Suffix(direction = Suffix.IMPORT)
    return m


def laed_opt_model():
    m = AbstractModel()
    m.N_g = Param(within=NonNegativeIntegers)
    m.N_t = Param(within=NonNegativeIntegers)
    m.N_T = Param(within=NonNegativeIntegers)
    m.N_s = Param(within=NonNegativeIntegers, default=0, initialize=0)

    m.G = RangeSet(1, m.N_g)
    m.T = RangeSet(1, m.N_t)
    m.S = RangeSet(1, m.N_s)

    m.Cost = Param(m.G)
    m.Capacity = Param(m.G)
    m.Ramp_lim = Param(m.G)
    m.Load = Param(m.T)
    m.Gen_init = Param(m.G)
    m.reserve_single = Param()

    m.E_cap = Param(m.S, default=0.0, initialize=lambda m, s: 0.0)
    m.P_ch_cap = Param(m.S, default=0.0, initialize=lambda m, s: 0.0)
    m.P_dis_cap = Param(m.S, default=0.0, initialize=lambda m, s: 0.0)
    m.SoC_init = Param(m.S, default=0.0, initialize=lambda m, s: 0.0)
    m.SoC_min = Param(m.S, default=0.0, initialize=lambda m, s: 0.0)
    m.SoC_max = Param(m.S, default=0.0, initialize=lambda m, s: 0.0)
    m.eta_c = Param(default=1.0, initialize=1.0)
    m.eta_d = Param(default=1.0, initialize=1.0)
    m.charge_cost = Param(m.S, default=0.0, initialize=lambda m, s: 0.0)
    m.discharge_cost = Param(m.S, default=0.0, initialize=lambda m, s: 0.0)
    m.delta_t = Param(default=1.0, initialize=1.0)

    m.P = Var(m.G, m.T, within=NonNegativeReals)
    m.Loadshed = Var(m.T, within=NonNegativeReals)
    m.Reserve = Var(m.G, m.T, within=NonNegativeReals)
    m.P_charge = Var(m.S, m.T, within=NonNegativeReals)
    m.P_discharge = Var(m.S, m.T, within=NonNegativeReals)
    m.SoC = Var(m.S, m.T, within=NonNegativeReals)

    def objective_rule(m):
        gen_cost = sum(sum(m.Cost[g] * m.P[g, t] for g in m.G) for t in m.T)
        shed_cost = cost_load * sum(m.Loadshed[t] for t in m.T)
        storage_cost = (
            sum(m.charge_cost[s] * m.P_charge[s, t] for s in m.S for t in m.T)
            + sum(m.discharge_cost[s] * m.P_discharge[s, t] for s in m.S for t in m.T)
        )
        return gen_cost + shed_cost + storage_cost
    m.obj = Objective(rule=objective_rule, sense=minimize)

    def capacity_rule(m, g, t):
        return m.P[g, t] + m.Reserve[g,t] <= m.Capacity[g]
    m.capacity_constraint = Constraint(m.G, m.T, rule=capacity_rule)

    def power_balance_rule(m, t):
        net_storage = sum(m.P_discharge[s, t] - m.P_charge[s, t] for s in m.S)
        return sum(m.P[g, t] for g in m.G) + net_storage == m.Load[t] - m.Loadshed[t]
    m.power_balance_constraint = Constraint(m.T, rule=power_balance_rule)

    def ramp_down_rule(m, g, t):
        if t == 1:
            return m.Gen_init[g] - m.P[g, 1] <= m.Ramp_lim[g]
        return m.P[g, t - 1] - m.P[g, t] <= m.Ramp_lim[g]
    m.ramp_down_constraint = Constraint(m.G, m.T, rule=ramp_down_rule)

    def ramp_up_rule(m, g, t):
        if t == 1:
            return m.P[g, 1] - m.Gen_init[g] <= m.Ramp_lim[g]
        return m.P[g, t] - m.P[g, t-1] <= m.Ramp_lim[g]
    m.ramp_up_constraint = Constraint(m.G, m.T, rule=ramp_up_rule)

    def reserve_rule(m, t):
        return sum(m.Reserve[g,t] for g in m.G) >= reserve_factor * m.Load[t]
    m.reserve_constraint = Constraint(m.T, rule=reserve_rule)

    def reserve_single_rule(m, g, t):
        return m.Reserve[g,t] <=  m.reserve_single * m.Capacity[g]
    m.reserve_single_constraint = Constraint(m.G, m.T, rule=reserve_single_rule)

    def storage_charge_cap_rule(m, s, t):
        return m.P_charge[s, t] <= m.P_ch_cap[s]
    m.storage_charge_cap_constraint = Constraint(m.S, m.T, rule=storage_charge_cap_rule)

    def storage_discharge_cap_rule(m, s, t):
        return m.P_discharge[s, t] <= m.P_dis_cap[s]
    m.storage_discharge_cap_constraint = Constraint(m.S, m.T, rule=storage_discharge_cap_rule)

    def storage_energy_bounds_rule(m, s, t):
        return pyo.inequality(m.SoC_min[s], m.SoC[s, t], m.SoC_max[s])
    m.storage_energy_bounds_constraint = Constraint(m.S, m.T, rule=storage_energy_bounds_rule)

    def storage_soc_balance_rule(m, s, t):
        dt = m.delta_t
        if t == 1:
            return m.SoC[s, t] == m.SoC_init[s] + m.eta_c * m.P_charge[s, t] * dt - (1 / m.eta_d) * m.P_discharge[s, t] * dt
        return m.SoC[s, t] == m.SoC[s, t-1] + m.eta_c * m.P_charge[s, t] * dt - (1 / m.eta_d) * m.P_discharge[s, t] * dt
    m.storage_soc_balance_constraint = Constraint(m.S, m.T, rule=storage_soc_balance_rule)

    def storage_no_simul_rule(m, s, t):
        cap = max(float(pyo.value(m.P_ch_cap[s])), float(pyo.value(m.P_dis_cap[s])))
        return m.P_charge[s, t] + m.P_discharge[s, t] <= cap
    m.storage_no_simul_constraint = Constraint(m.S, m.T, rule=storage_no_simul_rule)

    m.dual = Suffix(direction=Suffix.IMPORT)
    return m


# -------- Prices --------
def LMP_calculation(model):
    t_commit = 1
    tN = model.T.last()

    P_value = np.array([value(model.P[g]) for g in model.G], dtype=float)
    rup_value = np.array([value(model.Rampup[g, tN]) for g in model.G], dtype=float)
    rdw_value = np.array([value(model.Rampdown[g, tN]) for g in model.G], dtype=float)
    loadshed_value = value(model.Loadshed[t_commit])

    lam = model.dual.get(model.power_balance_constraint, 0.0)
    mu_rup = float(model.dual.get(model.ru_endpoint_constraint, 0.0)) if tN >= 2 else 0.0
    mu_rdw = float(model.dual.get(model.rd_endpoint_constraint, 0.0)) if tN >= 2 else 0.0

    LMP = abs(lam)
    TLMP = abs(lam)  - abs(mu_rup) + abs(mu_rdw)
    rup_price = abs(mu_rup)
    rdw_price = abs(mu_rdw)
    return P_value, loadshed_value, LMP, TLMP, rup_value, rup_price, rdw_value, rdw_price


def TLMP_calculation(model, N_g, N_t):
    H = model.T.last()
    P_value = np.array([[value(model.P[g, t]) for t in model.T] for g in model.G], dtype=float)
    R_value = np.array([[value(model.Reserve[g,t]) for t in model.T] for g in model.G], dtype=float)
    loadshed_value = np.array([value(model.Loadshed[t]) for t in model.T], dtype=float)
    la = np.array([model.dual.get(model.power_balance_constraint[t], 0.0) for t in model.T], dtype=float)
    mu_down = np.array([[model.dual.get(model.ramp_down_constraint[g, t], 0.0) for t in model.T] for g in model.G], dtype=float)
    mu_up = np.array([[model.dual.get(model.ramp_up_constraint[g, t], 0.0) for t in model.T] for g in model.G], dtype=float)
    R_price = np.abs(np.array([model.dual.get(model.reserve_constraint[t], 0.0) for t in model.T], dtype=float))

    TLMP_T = np.zeros((N_g, H), dtype=float)
    for gi, _ in enumerate(model.G):
        for t in range(1, H+1):
            t_idx = t - 1
            if t < H:
                mup_next = mu_up[gi, t_idx + 1]
                mdn_next = mu_down[gi, t_idx + 1]
            else:
                mup_next = 0.0
                mdn_next = 0.0
            
            TLMP_T[gi, t_idx] = (
                la[t_idx]
                + (mu_up[gi, t_idx] - mup_next)
                - (mu_down[gi, t_idx] - mdn_next)
            )
    return P_value, loadshed_value, TLMP_T, la, mu_down, mu_up, R_value, R_price


# -------- Deterministic rolling problems with storage --------
def ED_no_error_storage(
    data, N_g, N_t, N_T, load_factor, ramp_factor, solver
):
    n_steps = N_T - N_t + 1
    base = data.data()
    storage_inputs = _extract_storage_inputs(base)

    LMP      = np.zeros((N_g, n_steps))
    TLMP     = np.zeros((N_g, n_steps))
    P_LMP    = np.zeros((N_g, n_steps))
    rup_ED   = np.zeros((N_g, n_steps))
    rupp_Ed  = np.zeros((N_g, n_steps))
    rdw_ED   = np.zeros((N_g, n_steps))
    rdwp_Ed  = np.zeros((N_g, n_steps))
    Shed_ED  = np.zeros(n_steps)
    N_s      = storage_inputs["N_s"]
    SoC_ED   = np.zeros((N_s, n_steps)) if N_s > 0 else np.zeros((0, n_steps))
    Pch_ED   = np.zeros((N_s, n_steps)) if N_s > 0 else np.zeros((0, n_steps))
    Pdis_ED  = np.zeros((N_s, n_steps)) if N_s > 0 else np.zeros((0, n_steps))

    load_true = {t: base["Load"][t] * load_factor for t in base["Load"]}
    gen_init  = {g: base["Gen_init"][g] * load_factor for g in base["Gen_init"]}
    ramp_init = {g: base["Ramp_lim"][g] * ramp_factor for g in base["Ramp_lim"]}
    cap_init  = dict(base["Capacity"])
    cost_init = dict(base["Cost"])
    ramp_single = base["ramp_single"] if isinstance(base["ramp_single"], (int, float)) else base["ramp_single"][None]
    reserve_single = base["reserve_single"] if isinstance(base["reserve_single"], (int, float)) else base["reserve_single"][None]

    rped_model = rped_opt_model()
    gen_prev_current = dict(gen_init)
    soc_init_current = dict(storage_inputs["SoC_init"])

    for k in range(n_steps):
        T0 = k + 1
        H = N_t
        load_window = {tt: load_true[T0 + tt - 1] for tt in range(1, H + 1)}

        instance_data = {
            None: {
                "N_g": {None: N_g},
                "N_t": {None: H},
                "Cost": cost_init,
                "Capacity": cap_init,
                "Ramp_lim": ramp_init,
                "Gen_prev": gen_prev_current,
                "Load": load_window,
                "ramp_single": {None: ramp_single},
                "reserve_single": {None: reserve_single},
                "N_s": {None: storage_inputs["N_s"]},
                "E_cap": storage_inputs["E_cap"],
                "P_ch_cap": storage_inputs["P_ch_cap"],
                "P_dis_cap": storage_inputs["P_dis_cap"],
                "SoC_init": soc_init_current,
                "SoC_min": storage_inputs["SoC_min"],
                "SoC_max": storage_inputs["SoC_max"],
                "charge_cost": storage_inputs["charge_cost"],
                "discharge_cost": storage_inputs["discharge_cost"],
                "eta_c": {None: storage_inputs["eta_c"]},
                "eta_d": {None: storage_inputs["eta_d"]},
                "delta_t": {None: storage_inputs["delta_t"]},
            }
        }

        rped = rped_model.create_instance(data=instance_data)
        results = solver.solve(rped, tee=False, load_solutions=True)
        rped.solutions.load_from(results)

        P_ed, Shed_ed, LMP_ed, TLMP_ed, rup_ed, rupp_ed, rdw_ed, rdwp_ed = LMP_calculation(rped)

        LMP[:, k]     = LMP_ed
        TLMP[:, k]    = TLMP_ed
        P_LMP[:, k]   = P_ed
        rup_ED[:, k]  = rup_ed
        rupp_Ed[:, k] = rupp_ed
        rdw_ED[:, k]  = rdw_ed
        rdwp_Ed[:, k] = rdwp_ed
        Shed_ED[k]    = float(Shed_ed)
        if N_s > 0:
            for s in range(1, N_s + 1):
                s_idx = s - 1
                SoC_ED[s_idx, k]  = float(pyo.value(rped.SoC[s, 1]))
                Pch_ED[s_idx, k]  = float(pyo.value(rped.P_charge[s, 1]))
                Pdis_ED[s_idx, k] = float(pyo.value(rped.P_discharge[s, 1]))

        gen_prev_current = {g: float(P_ed[g - 1]) for g in range(1, N_g + 1)}
        if N_s > 0:
            soc_init_current = {s: float(pyo.value(rped.SoC[s, rped.T.last()])) for s in range(1, N_s + 1)}

    return P_LMP, Shed_ED, LMP, TLMP, rup_ED, rupp_Ed, rdw_ED, rdwp_Ed, SoC_ED, Pch_ED, Pdis_ED


def LAED_no_error_storage(
    data, N_g, N_t, N_T, load_factor, ramp_factor, solver
):
    n_steps = N_T - N_t + 1
    base = data.data()
    storage_inputs = _extract_storage_inputs(base)

    TLMP = np.zeros((N_g, n_steps))
    LLMP = np.zeros((N_g, n_steps))
    P_LAED = np.zeros((N_g, n_steps))
    R_LAED = np.zeros((N_g, n_steps))
    RP_LAED = np.zeros((N_g, n_steps))
    Shed_LAED = np.zeros(n_steps)
    N_s = storage_inputs["N_s"]
    SoC_LAED  = np.zeros((N_s, n_steps)) if N_s > 0 else np.zeros((0, n_steps))
    Pch_LAED  = np.zeros((N_s, n_steps)) if N_s > 0 else np.zeros((0, n_steps))
    Pdis_LAED = np.zeros((N_s, n_steps)) if N_s > 0 else np.zeros((0, n_steps))

    load_true = {t: base["Load"][t] * load_factor for t in base["Load"]}
    gen_init  = {g: base["Gen_init"][g] * load_factor for g in base["Gen_init"]}
    ramp_init = {g: base["Ramp_lim"][g] * ramp_factor for g in base["Ramp_lim"]}
    cap_init  = dict(base["Capacity"])
    cost_init = dict(base["Cost"])
    reserve_single = base["reserve_single"] if isinstance(base["reserve_single"], (int, float)) else base["reserve_single"][None]

    model_laed = laed_opt_model()
    gen_init_current = dict(gen_init)
    soc_init_current = dict(storage_inputs["SoC_init"])

    for k in range(n_steps):
        T0 = k + 1
        load_window = {tt: load_true[T0 + tt - 1] for tt in range(1, N_t + 1)}

        instance_data = {
            None: {
                "N_g": {None: N_g},
                "N_t": {None: N_t},
                "N_T": {None: N_T},
                "Cost": cost_init,
                "Capacity": cap_init,
                "Ramp_lim": ramp_init,
                "Load": load_window,
                "Gen_init": gen_init_current,
                "reserve_single": {None: reserve_single},
                "N_s": {None: storage_inputs["N_s"]},
                "E_cap": storage_inputs["E_cap"],
                "P_ch_cap": storage_inputs["P_ch_cap"],
                "P_dis_cap": storage_inputs["P_dis_cap"],
                "SoC_init": soc_init_current,
                "SoC_min": storage_inputs["SoC_min"],
                "SoC_max": storage_inputs["SoC_max"],
                "charge_cost": storage_inputs["charge_cost"],
                "discharge_cost": storage_inputs["discharge_cost"],
                "eta_c": {None: storage_inputs["eta_c"]},
                "eta_d": {None: storage_inputs["eta_d"]},
                "delta_t": {None: storage_inputs["delta_t"]},
            }
        }

        laed = model_laed.create_instance(data=instance_data)
        results = solver.solve(laed, tee=False, load_solutions=True)
        laed.solutions.load_from(results)

        P_laed, Shed_laed, TLMP_T, LLMP_T, _, _, R_laed, Rp_laed = TLMP_calculation(laed, N_g, N_t)

        P_LAED[:, k] = P_laed[:, 0]
        R_LAED[:, k] = R_laed[:, 0]
        TLMP[:, k]   = TLMP_T[:, 0]
        LLMP[:, k]   = float(LLMP_T[0]) * np.ones(N_g)
        RP_LAED[:, k]= float(Rp_laed[0]) * np.ones(N_g)
        Shed_LAED[k] = float(Shed_laed[0])
        if N_s > 0:
            for s in range(1, N_s + 1):
                s_idx = s - 1
                SoC_LAED[s_idx, k]  = float(pyo.value(laed.SoC[s, 1]))
                Pch_LAED[s_idx, k]  = float(pyo.value(laed.P_charge[s, 1]))
                Pdis_LAED[s_idx, k] = float(pyo.value(laed.P_discharge[s, 1]))

        gen_init_current = {g: float(P_laed[g - 1, 0]) for g in range(1, N_g + 1)}
        if N_s > 0:
            soc_init_current = {s: float(pyo.value(laed.SoC[s, 1])) for s in range(1, N_s + 1)}

    return P_LAED, Shed_LAED, TLMP, LLMP, R_LAED, RP_LAED, SoC_LAED, Pch_LAED, Pdis_LAED


# -------- Main script (deterministic) --------
if __name__=="__main__":
    solver = SolverFactory("gurobi_direct")
    solver.options['OutputFlag'] = 0
    load_factor = 1.0
    reserve_factor = 0
    ramp_factor = 0.1
    cost_load = 1e10
    cost_curtailed = 0
    case_name = 'toy_data_storage.dat'

    data = DataPortal()
    data.load(filename=case_name)

    seed = 42
    rng = np.random.default_rng(seed)  # unused but kept for parity

    file_path = 'MISO_Projection.json'
    with open(file_path, 'r') as f:
        interpolated_data = json.load(f)

    ref_cap=350
    Aug_2032_ori = interpolated_data['2032_Aug']
    load_scale_2032 = ref_cap/(sum(Aug_2032_ori.values())/len(Aug_2032_ori))
    Aug_2032 = {int(key): value*load_scale_2032 for key, value in Aug_2032_ori.items()}
    data.data()["Load"] = Aug_2032
    # The projection is 288 points at 5-minute resolution (24h). Ensure storage SoC dynamics
    # use the correct timestep (hours) instead of whatever is in the .dat file.
    if int(len(Aug_2032_ori)) == 288 and "delta_t" in data.data():
        data.data()["delta_t"][None] = 1.0 / 12.0
    if "N_T" in data.data():
        data.data()["N_T"][None] = int(len(Aug_2032_ori))

    nts = np.linspace(1, 19, 19)
    laed_sheds = []
    rp_sheds = []

    for i in tqdm(range(1,20)):
        data.data()["N_t"][None] = int(i)
        N_g = data.data()['N_g'][None]
        N_t = data.data()['N_t'][None]
        N_T = len(Aug_2032_ori)

        if N_g != 2:
            Load_ini = data.data()['Load'][1]
            model_ini = AbstractModel()
            model_ini.N_g = Param(within=NonNegativeIntegers)
            model_ini.G = RangeSet(1, model_ini.N_g)
            model_ini.Cost = Param(model_ini.G)
            model_ini.Capacity = Param(model_ini.G)
            model_ini.reserve_single = Param()
            model_ini.P = Var(model_ini.G,  within=NonNegativeReals)
            model_ini.Reserve = Var(model_ini.G, within=NonNegativeReals)
            def objective_rule(model):
                return sum(model.Cost[g] * model.P[g] for g in model.G) 
            model_ini.obj = Objective(rule=objective_rule, sense=minimize)
            def power_balance_rule(model):
                return sum(model.P[g] for g in model.G) == Load_ini 
            model_ini.power_balance_constraint = Constraint(rule=power_balance_rule)
            def capacity_rule(model,g):
                return model.P[g] + model.Reserve[g] <= model.Capacity[g]
            model_ini.capacity_constraint = Constraint(model_ini.G, rule=capacity_rule)
            def reserve_rule(model):
                return sum(model.Reserve[g] for g in model.G) >= reserve_factor * Load_ini
            model_ini.reserve_constraint = Constraint(rule=reserve_rule)
            def reserve_single_rule(model, g):
                return model.Reserve[g] <= model.reserve_single * model.Capacity[g]
            model_ini.reserve_single_constraint = Constraint(model_ini.G,  rule=reserve_single_rule)

            ed_ini = model_ini.create_instance(data)
            solver.solve(ed_ini, tee=False)
            data.data()['Gen_init'] = {g: ed_ini.P[g].value for g in ed_ini.G}
        else:
            Gen1_ini = np.min([data.data()['Capacity'][1],data.data()['Load'][1]])
            data.data()['Gen_init'] = {1: Gen1_ini, 2: np.min([data.data()['Capacity'][2], data.data()['Load'][1]- Gen1_ini])}

        data_laed = copy.deepcopy(data)
        data_ed = copy.deepcopy(data)
            
        P_ED, Shed_ED, LMP, TLMP, rup_ED, rupp_Ed, rdw_ED, rdwp_Ed, SoC_ED, Pch_ED, Pdis_ED = ED_no_error_storage(data_ed, N_g, N_t, N_T, load_factor, ramp_factor, solver)
        P_LAED, Shed_LAED, TLMP, LLMP, R_LAED, RP_LAED, SoC_LAED, Pch_LAED, Pdis_LAED = LAED_no_error_storage(data_laed, N_g, N_t, N_T, load_factor, ramp_factor, solver)

        laed_sheds.append(np.sum(Shed_LAED)/12)
        rp_sheds.append(np.sum(Shed_ED)/12)

    diffs = [laed_sheds[i] - rp_sheds[i] for i in range(len(laed_sheds))]
    print(diffs)

    plt.figure()
    plt.plot(nts, laed_sheds, label="LAED")
    plt.plot(nts, rp_sheds, label="RP")
    plt.xlabel('Window Size')
    plt.ylabel("Load Shedding")
    plt.xticks(nts.astype(int))
    plt.legend()
    plt.show()

    if SoC_LAED.size > 0:
        times = np.arange(1, SoC_LAED.shape[1] + 1)
        plt.figure()
        for s_idx in range(SoC_LAED.shape[0]):
            plt.plot(times, SoC_LAED[s_idx], label=f"SoC LAED s{ s_idx+1 }")
        plt.xlabel("Physical time index")
        plt.ylabel("State of Charge (MWh)")
        plt.title("LAED storage SoC (last sweep run)")
        plt.legend()
        plt.show()

        plt.figure()
        for s_idx in range(SoC_LAED.shape[0]):
            plt.plot(times, Pch_LAED[s_idx], label=f"Charge LAED s{ s_idx+1 }")
            plt.plot(times, Pdis_LAED[s_idx], linestyle="--", label=f"Discharge LAED s{ s_idx+1 }")
        plt.xlabel("Physical time index")
        plt.ylabel("Power (MW)")
        plt.title("LAED storage charge/discharge (last sweep run)")
        plt.legend()
        plt.show()

    if SoC_ED.size > 0:
        times = np.arange(1, SoC_ED.shape[1] + 1)
        plt.figure()
        for s_idx in range(SoC_ED.shape[0]):
            plt.plot(times, SoC_ED[s_idx], label=f"SoC RP s{ s_idx+1 }")
        plt.xlabel("Physical time index")
        plt.ylabel("State of Charge (MWh)")
        plt.title("RP storage SoC (last sweep run)")
        plt.legend()
        plt.show()

        plt.figure()
        for s_idx in range(SoC_ED.shape[0]):
            plt.plot(times, Pch_ED[s_idx], label=f"Charge RP s{ s_idx+1 }")
            plt.plot(times, Pdis_ED[s_idx], linestyle="--", label=f"Discharge RP s{ s_idx+1 }")
        plt.xlabel("Physical time index")
        plt.ylabel("Power (MW)")
        plt.title("RP storage charge/discharge (last sweep run)")
        plt.legend()
        plt.show()
