import pyomo.environ as pyo
from pyomo.environ import (
    AbstractModel, Param, RangeSet, Var, Constraint, Objective, Suffix, minimize, DataPortal,
    NonNegativeIntegers, NonNegativeReals, value, SolverStatus, TerminationCondition, SolverFactory
)
import numpy as np
import copy
import matplotlib.pyplot as plt
import json
from tqdm import tqdm

# Default values (overridden in __main__ when running as a script)
cost_load = 3500.0
reserve_factor = 0.0


def _unit_variance_innovation(error_type: str, rng: np.random.Generator, **kwargs) -> float:
    """Return random draw z with E[z] = 0, Var[z] = 1 for requested distribution"""
    et = error_type.lower()

    if et in ('gaussian', 'normal'):
        return float(rng.standard_normal())
    
    elif et in ("laplace", "double_exponential", "double-exponential"):
        return float(rng.laplace(loc=0.0, scale=1.0 / np.sqrt(2.0)))
    
    elif et in ("student_t", "student-t", "t"):
        nu = float(kwargs.get("nu", 4))
        if nu <= 2:
            raise ValueError("Student-t requires nu > 2 for finite variance")
        z_raw = rng.standard_t(df=nu)
        return float(z_raw / np.sqrt(nu / (nu - 2.0)))
    
    raise ValueError(f"Unknown error_type: {error_type}. Use 'gaussian', 'laplace' or 'student-t'")
        

def correlated_error_update(prev_err, std_h, rho, rng, error_type = "gaussian", **error_kwargs):
    rho = float(rho)
    if not (0.0 <= rho < 1.0):
        raise ValueError("rho must be [0, 1).")
    
    innovation_scale = np.sqrt(1.0 - rho**2)
    z = _unit_variance_innovation(error_type = error_type, rng = rng, **error_kwargs)
    return float(rho*prev_err + innovation_scale*std_h*z)


def _extract_storage_inputs(base: dict):
    """Return storage parameter dictionaries with safe defaults if storage is absent."""
    n_s = 0
    if "N_s" in base:
        val = base["N_s"]
        n_s = int(val.get(None, 0) if isinstance(val, dict) else val)
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
    if "SoC_max" in base:
        SoC_max = _by_s("SoC_max", 0.0)
    else:
        SoC_max = {s: E_cap.get(s, 0.0) for s in S_idx}

    charge_cost = _by_s("charge_cost", 0.0)
    discharge_cost = _by_s("discharge_cost", 0.0)

    eta_c = 1.0
    eta_d = 1.0
    delta_t = 1.0
    if "eta_c" in base:
        eta_c = float(base["eta_c"].get(None, 1.0) if isinstance(base["eta_c"], dict) else base["eta_c"])
    if "eta_d" in base:
        eta_d = float(base["eta_d"].get(None, 1.0) if isinstance(base["eta_d"], dict) else base["eta_d"])
    if "delta_t" in base:
        delta_t = float(base["delta_t"].get(None, 1.0) if isinstance(base["delta_t"], dict) else base["delta_t"])

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


import pyomo.environ as pyo
from pyomo.environ import (
    AbstractModel, Param, RangeSet, Var, Constraint, Objective, Suffix, minimize,
    NonNegativeIntegers, NonNegativeReals, Binary
)

# Uses your globals
# cost_load, reserve_factor

def rped_opt_model():
    """
    RPED with storage:
      - Executed (physical) storage only at t=1 via P_charge[s,1], P_discharge[s,1]
      - Ramp-product endpoint uses capability-only variables P_charge_end[s], P_discharge_end[s]
        that DO NOT affect SoC dynamics (no execution), but must be feasible given SoC at t=1.
      - Exact non-simultaneous charge/discharge enforced with binaries (MILP).
    """
    m = AbstractModel()

    # sizes / sets
    m.N_g = Param(within=NonNegativeIntegers)
    m.N_t = Param(within=NonNegativeIntegers)
    m.N_s = Param(within=NonNegativeIntegers, default=0, initialize=0)

    m.G = RangeSet(1, m.N_g)
    m.T = RangeSet(1, m.N_t)
    m.S = RangeSet(1, m.N_s)

    # parameters
    m.Cost = Param(m.G)
    m.Capacity = Param(m.G)
    m.Ramp_lim = Param(m.G)
    m.Load = Param(m.T)
    m.Gen_prev = Param(m.G)
    m.ramp_single = Param()
    m.reserve_single = Param()

    # storage params
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
    m.delta_t = Param(default=1.0, initialize=1.0)  # hours

    # decision variables
    m.P = Var(m.G, within=NonNegativeReals)
    m.Rampup = Var(m.G, m.T, within=NonNegativeReals)
    m.Rampdown = Var(m.G, m.T, within=NonNegativeReals)
    m.Loadshed = Var(m.T, within=NonNegativeReals)

    # EXECUTED storage at t=1 (keep indexed by T for convenience, but we force t!=1 to 0)
    m.P_charge = Var(m.S, m.T, within=NonNegativeReals)
    m.P_discharge = Var(m.S, m.T, within=NonNegativeReals)
    m.SoC = Var(m.S, m.T, within=NonNegativeReals)

    # CAPABILITY-ONLY endpoint storage (NOT executed)
    m.P_charge_end = Var(m.S, within=NonNegativeReals)
    m.P_discharge_end = Var(m.S, within=NonNegativeReals)

    # binaries to prevent simultaneity (executed vs endpoint capability)
    m.u_exec = Var(m.S, within=Binary)  # 1 => charge allowed at t=1, 0 => discharge allowed at t=1
    m.u_end  = Var(m.S, within=Binary)  # 1 => charge allowed at endpoint, 0 => discharge allowed at endpoint

    # objective
    def objective_rule(m):
        gen_cost = sum(m.Cost[g] * m.P[g] for g in m.G)
        shed_cost = cost_load * sum(m.Loadshed[t] for t in m.T)
        # Only executed storage at t=1 appears in objective (consistent with "capability-only" endpoint)
        storage_cost = sum(
            m.charge_cost[s] * m.P_charge[s, 1] + m.discharge_cost[s] * m.P_discharge[s, 1]
            for s in m.S
        )
        return gen_cost + shed_cost + storage_cost
    m.obj = Objective(rule=objective_rule, sense=minimize)

    # forbid shedding in non-committed intervals
    # def future_shed_zero_rule(m, t):
    #     if t == 1:
    #         return Constraint.Skip
    #     return m.Loadshed[t] == 0.0
    # m.future_shed_zero_constraint = Constraint(m.T, rule=future_shed_zero_rule)

    # generator capacity wrt ramp product quantities
    def capacity_up_rule(m, g, t):
        return m.P[g] + m.Rampup[g, t] <= m.Capacity[g]
    m.capacity_up_constraint = Constraint(m.G, m.T, rule=capacity_up_rule)

    def capacity_down_rule(m, g, t):
        return m.P[g] - m.Rampdown[g, t] >= 0
    m.capacity_down_constraint = Constraint(m.G, m.T, rule=capacity_down_rule)

    # power balance at commitment interval t=1 (executed storage only)
    def power_balance_rule(m):
        net_storage_t1 = sum(m.P_discharge[s, 1] - m.P_charge[s, 1] for s in m.S)
        return sum(m.P[g] for g in m.G) + net_storage_t1 == m.Load[1] - m.Loadshed[1]
    m.power_balance_constraint = Constraint(rule=power_balance_rule)

    # generator ramp from previous realized dispatch
    def ramp_down_rule(m, g):
        return m.P[g] - m.Gen_prev[g] >= -m.Ramp_lim[g]
    m.ramp_down_constraint = Constraint(m.G, rule=ramp_down_rule)

    def ramp_up_rule(m, g):
        return m.P[g] - m.Gen_prev[g] <= m.Ramp_lim[g]
    m.ramp_up_constraint = Constraint(m.G, rule=ramp_up_rule)

    # ramp-product variables only active at t=2 (your current convention)
    def rampup_window_rule(m, g, t):
        if t == 1:
            return m.Rampup[g, t] <= 0.0
        if t == 2:
            return m.Rampup[g, t] <= m.Ramp_lim[g]
        return m.Rampup[g, t] <= 0.0
    m.rampup_window_constraint = Constraint(m.G, m.T, rule=rampup_window_rule)

    def rampdown_window_rule(m, g, t):
        if t == 1:
            return m.Rampdown[g, t] <= 0.0
        if t == 2:
            return m.Rampdown[g, t] <= m.Ramp_lim[g]
        return m.Rampdown[g, t] <= 0.0
    m.rampdown_window_constraint = Constraint(m.G, m.T, rule=rampdown_window_rule)

    # -------------------------
    # STORAGE: executed actions only at t=1
    # -------------------------

    # power caps on executed vars
    def storage_charge_cap_rule(m, s, t):
        return m.P_charge[s, t] <= m.P_ch_cap[s]
    m.storage_charge_cap_constraint = Constraint(m.S, m.T, rule=storage_charge_cap_rule)

    def storage_discharge_cap_rule(m, s, t):
        return m.P_discharge[s, t] <= m.P_dis_cap[s]
    m.storage_discharge_cap_constraint = Constraint(m.S, m.T, rule=storage_discharge_cap_rule)

    # force executed charge/discharge to 0 for t != 1
    def storage_exec_only_t1_charge_rule(m, s, t):
        if t == 1:
            return Constraint.Skip
        return m.P_charge[s, t] == 0.0
    m.storage_exec_only_t1_charge = Constraint(m.S, m.T, rule=storage_exec_only_t1_charge_rule)

    def storage_exec_only_t1_discharge_rule(m, s, t):
        if t == 1:
            return Constraint.Skip
        return m.P_discharge[s, t] == 0.0
    m.storage_exec_only_t1_discharge = Constraint(m.S, m.T, rule=storage_exec_only_t1_discharge_rule)

    # exact non-simultaneous at t=1 (executed)
    def storage_exec_no_simul_charge(m, s):
        return m.P_charge[s, 1] <= m.P_ch_cap[s] * m.u_exec[s]
    m.storage_exec_no_simul_charge = Constraint(m.S, rule=storage_exec_no_simul_charge)

    def storage_exec_no_simul_discharge(m, s):
        return m.P_discharge[s, 1] <= m.P_dis_cap[s] * (1 - m.u_exec[s])
    m.storage_exec_no_simul_discharge = Constraint(m.S, rule=storage_exec_no_simul_discharge)

    # SoC bounds (apply to all t, but we will freeze SoC across t anyway)
    def storage_energy_bounds_rule(m, s, t):
        return pyo.inequality(m.SoC_min[s], m.SoC[s, t], m.SoC_max[s])
    m.storage_energy_bounds_constraint = Constraint(m.S, m.T, rule=storage_energy_bounds_rule)

    # SoC update ONLY at t=1 (executed physics)
    def storage_soc_balance_t1_rule(m, s):
        dt = m.delta_t
        return m.SoC[s, 1] == (
            m.SoC_init[s]
            + m.eta_c * m.P_charge[s, 1] * dt
            - (1 / m.eta_d) * m.P_discharge[s, 1] * dt
        )
    m.storage_soc_balance_t1 = Constraint(m.S, rule=storage_soc_balance_t1_rule)

    # Freeze SoC across the (advisory/capability) horizon: SoC[t] = SoC[1] for t>1
    def storage_soc_freeze_rule(m, s, t):
        if t == 1:
            return Constraint.Skip
        return m.SoC[s, t] == m.SoC[s, 1]
    m.storage_soc_freeze = Constraint(m.S, m.T, rule=storage_soc_freeze_rule)

    # -------------------------
    # STORAGE: endpoint capability-only variables
    # -------------------------

    # endpoint power caps
    def storage_end_charge_cap(m, s):
        return m.P_charge_end[s] <= m.P_ch_cap[s] * m.u_end[s]
    m.storage_end_charge_cap = Constraint(m.S, rule=storage_end_charge_cap)

    def storage_end_discharge_cap(m, s):
        return m.P_discharge_end[s] <= m.P_dis_cap[s] * (1 - m.u_end[s])
    m.storage_end_discharge_cap = Constraint(m.S, rule=storage_end_discharge_cap)

    # endpoint capability must be feasible wrt SoC at t=1 for ONE interval of length delta_t
    # (discrete endpoint capability consistency condition)
    def storage_end_energy_feas_charge(m, s):
        dt = m.delta_t
        return m.SoC[s, 1] + m.eta_c * m.P_charge_end[s] * dt <= m.SoC_max[s]
    m.storage_end_energy_feas_charge = Constraint(m.S, rule=storage_end_energy_feas_charge)

    def storage_end_energy_feas_discharge(m, s):
        dt = m.delta_t
        return m.SoC[s, 1] - (1 / m.eta_d) * m.P_discharge_end[s] * dt >= m.SoC_min[s]
    m.storage_end_energy_feas_discharge = Constraint(m.S, rule=storage_end_energy_feas_discharge)

    # -------------------------
    # Ramp endpoint constraints using storage NET injection change from t=1 -> t=end (capability)
    # -------------------------
    def ru_endpoint_rule(m):
        t1 = 1
        tN = m.T.last()

        storage_net_t1 = sum(m.P_discharge[s, t1] - m.P_charge[s, t1] for s in m.S)
        storage_net_tN = sum(m.P_discharge_end[s] - m.P_charge_end[s] for s in m.S)

        return (
            sum(m.Rampup[g, tN] for g in m.G)
            + (storage_net_tN - storage_net_t1)
            >= (m.Load[tN] - m.Load[t1])
        )
    m.ru_endpoint_constraint = Constraint(rule=ru_endpoint_rule)

    def rd_endpoint_rule(m):
        t1 = 1
        tN = m.T.last()

        storage_net_t1 = sum(m.P_discharge[s, t1] - m.P_charge[s, t1] for s in m.S)
        storage_net_tN = sum(m.P_discharge_end[s] - m.P_charge_end[s] for s in m.S)

        return (
            sum(m.Rampdown[g, tN] for g in m.G)
            - (storage_net_tN - storage_net_t1)
            >= (m.Load[t1] - m.Load[tN])
        )
    m.rd_endpoint_constraint = Constraint(rule=rd_endpoint_rule)

    # dual suffix
    m.dual = Suffix(direction=Suffix.IMPORT)
    return m



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

    #mu_rup = model.dual.get(model.rampup_reserve_constraint[t_commit], 0.0)
    #mu_rdw = model.dual.get(model.rampdw_reserve_constraint[t_commit], 0.0)

    # Your custom “LMP” combination (note: abs() is not standard)
    LMP = abs(lam)
    TLMP = abs(lam)  - abs(mu_rup) + abs(mu_rdw)
    rup_price = abs(mu_rup)
    rdw_price = abs(mu_rdw)

    return P_value, loadshed_value, LMP, TLMP, rup_value, rup_price, rdw_value, rdw_price

def ED_with_error(
    data, N_g, N_t, N_T, load_factor, ramp_factor, solver,
    sigma_rel,    # relative 1-step error scale (tunable)
    rho,           # correlation of forecast revisions across rolling windows
    rng,
    error_type="gaussian",
    error_kwargs=None,
    clip_nonnegative=True
):
    
    if error_kwargs is None:
        error_kwargs = {}

    # This RP/ED model is implemented as a 10-minute ramp product.
    # Match laed_rp_analysis.py by using a 2-interval window (t=1 commit, t=2 ramp product endpoint).
    if int(N_t) != 2:
        raise ValueError("ED_with_error expects N_t=2 (10-minute ramp product).")

    n_steps = N_T - N_t + 1

    base = data.data()
    storage_inputs = _extract_storage_inputs(base)
    N_s = storage_inputs["N_s"]

    LMP      = np.zeros((N_g, n_steps))
    TLMP     = np.zeros((N_g, n_steps))
    P_LMP    = np.zeros((N_g, n_steps))
    rup_ED   = np.zeros((N_g, n_steps))
    rupp_Ed  = np.zeros((N_g, n_steps))
    rdw_ED   = np.zeros((N_g, n_steps))
    rdwp_Ed  = np.zeros((N_g, n_steps))
    Shed_ED  = np.zeros(n_steps)
    SoC_ED   = np.zeros((N_s, n_steps)) if N_s > 0 else np.zeros((0, n_steps))
    Pch_ED   = np.zeros((N_s, n_steps)) if N_s > 0 else np.zeros((0, n_steps))
    Pdis_ED  = np.zeros((N_s, n_steps)) if N_s > 0 else np.zeros((0, n_steps))

    # True underlying load (scaled) for each physical time in 1..N_T
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

    # Persistent forecast error state per PHYSICAL time index t (1..N_T)
    err_state = {t: 0.0 for t in range(1, N_T + 1)}

    for k in range(n_steps):
        T0 = k + 1
        H = N_t

        # At current time, you "observe" load, so forecast error is 0 at lead=0
        err_state[T0] = 0.0

        # windowed forecast load reindexed to 1..H
        load_window = {}

        for tt in range(1, H + 1):
            t_phys = T0 + tt - 1
            h = tt - 1  # lead time within window

            if h == 0:
                e = 0.0
                err_state[t_phys] = 0.0
            else:
                # Var ∝ h  => Std ∝ sqrt(h)
                std_h = sigma_rel * abs(load_true[t_phys]) * np.sqrt(h)

                e = correlated_error_update(
                    prev_err = err_state[t_phys],
                    std_h = std_h,
                    rho = rho,
                    rng=rng,
                    error_type=error_type,
                    **error_kwargs
                )

                err_state[t_phys] = e

            Lhat = float(load_true[t_phys]) + e
            if clip_nonnegative:
                Lhat = max(0.0, Lhat)

            load_window[tt] = float(Lhat)

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

        results = solver.solve(rped, tee=False, load_solutions=False)
        term = str(results.solver.termination_condition).lower()
        status = str(results.solver.status).lower()
        if term != "optimal" or status != "ok":
            raise RuntimeError(f"RPED infeasible/unknown at window {k+1}: status={status}, term={term}")
        rped.solutions.load_from(results)

        P_ed, Shed_ed, LMP_ed, TLMP_ed, rup_ed, rupp_ed, rdw_ed, rdwp_ed = LMP_calculation(rped)
        # optional: record storage trajectories per window by importing storage.py helpers

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

        # update initial condition for next window (explicit by generator index)
        gen_prev_current = {g: float(P_ed[g - 1]) for g in range(1, N_g + 1)}
        if storage_inputs["N_s"] > 0:
            # Rolling implementation commits/records interval t=1, so roll SoC forward from t=1.
            soc_init_current = {s: float(pyo.value(rped.SoC[s, 1])) for s in range(1, storage_inputs["N_s"] + 1)}

    return P_LMP, Shed_ED, LMP, TLMP, rup_ED, rupp_Ed, rdw_ED, rdwp_Ed, SoC_ED, Pch_ED, Pdis_ED


def laed_opt_model():
    """
    LAED storage (executed across the look-ahead horizon).
    Changes vs your version:
      - exact non-simultaneous charge/discharge enforced with binaries per (s,t)
      - SoC balance unchanged and correct
    """
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

    # storage params
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

    # vars
    m.P = Var(m.G, m.T, within=NonNegativeReals)
    m.Loadshed = Var(m.T, within=NonNegativeReals)
    m.Reserve = Var(m.G, m.T, within=NonNegativeReals)

    m.P_charge = Var(m.S, m.T, within=NonNegativeReals)
    m.P_discharge = Var(m.S, m.T, within=NonNegativeReals)
    m.SoC = Var(m.S, m.T, within=NonNegativeReals)

    # binaries for exact non-simultaneous charge/discharge
    m.u_stor = Var(m.S, m.T, within=Binary)  # 1 => charge allowed, 0 => discharge allowed

    def objective_rule(m):
        gen_cost = sum(m.Cost[g] * m.P[g, t] for g in m.G for t in m.T)
        shed_cost = cost_load * sum(m.Loadshed[t] for t in m.T)
        storage_cost = (
            sum(m.charge_cost[s] * m.P_charge[s, t] for s in m.S for t in m.T)
            + sum(m.discharge_cost[s] * m.P_discharge[s, t] for s in m.S for t in m.T)
        )
        return gen_cost + shed_cost + storage_cost
    m.obj = Objective(rule=objective_rule, sense=minimize)

    def capacity_rule(m, g, t):
        return m.P[g, t] + m.Reserve[g, t] <= m.Capacity[g]
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
        return m.P[g, t] - m.P[g, t - 1] <= m.Ramp_lim[g]
    m.ramp_up_constraint = Constraint(m.G, m.T, rule=ramp_up_rule)

    def reserve_rule(m, t):
        return sum(m.Reserve[g, t] for g in m.G) >= reserve_factor * m.Load[t]
    m.reserve_constraint = Constraint(m.T, rule=reserve_rule)

    def reserve_single_rule(m, g, t):
        return m.Reserve[g, t] <= m.reserve_single * m.Capacity[g]
    m.reserve_single_constraint = Constraint(m.G, m.T, rule=reserve_single_rule)

    def storage_charge_cap_rule(m, s, t):
        return m.P_charge[s, t] <= m.P_ch_cap[s]
    m.storage_charge_cap_constraint = Constraint(m.S, m.T, rule=storage_charge_cap_rule)

    def storage_discharge_cap_rule(m, s, t):
        return m.P_discharge[s, t] <= m.P_dis_cap[s]
    m.storage_discharge_cap_constraint = Constraint(m.S, m.T, rule=storage_discharge_cap_rule)

    # exact no-simultaneous using binaries
    def storage_no_simul_charge(m, s, t):
        return m.P_charge[s, t] <= m.P_ch_cap[s] * m.u_stor[s, t]
    m.storage_no_simul_charge = Constraint(m.S, m.T, rule=storage_no_simul_charge)

    def storage_no_simul_discharge(m, s, t):
        return m.P_discharge[s, t] <= m.P_dis_cap[s] * (1 - m.u_stor[s, t])
    m.storage_no_simul_discharge = Constraint(m.S, m.T, rule=storage_no_simul_discharge)

    def storage_energy_bounds_rule(m, s, t):
        return pyo.inequality(m.SoC_min[s], m.SoC[s, t], m.SoC_max[s])
    m.storage_energy_bounds_constraint = Constraint(m.S, m.T, rule=storage_energy_bounds_rule)

    def storage_soc_balance_rule(m, s, t):
        dt = m.delta_t
        if t == 1:
            return m.SoC[s, t] == (
                m.SoC_init[s]
                + m.eta_c * m.P_charge[s, t] * dt
                - (1 / m.eta_d) * m.P_discharge[s, t] * dt
            )
        return m.SoC[s, t] == (
            m.SoC[s, t - 1]
            + m.eta_c * m.P_charge[s, t] * dt
            - (1 / m.eta_d) * m.P_discharge[s, t] * dt
        )
    m.storage_soc_balance_constraint = Constraint(m.S, m.T, rule=storage_soc_balance_rule)

    m.dual = Suffix(direction=Suffix.IMPORT)
    return m

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


def LAED_with_error(
    data, N_g, N_t, N_T, load_factor, ramp_factor, solver,
    sigma_rel,   # relative 1-step error scale (tunable)
    rho,          # correlation of forecast revisions across rolling windows
    rng,
    error_type = 'gaussian', 
    error_kwargs=None,
    clip_nonnegative=True,
):
    """
    LAED rolling window with correlated Gaussian load forecast error.

    Error model (for a physical time index t_phys at window start T0):
      lead h = t_phys - T0  (h=0 is 'now')
      target variance decreases linearly as h->0:
        Var(e | lead=h) = (sigma_rel * Load_true[t_phys])^2 * h
        => Std = sigma_rel * Load_true[t_phys] * sqrt(h)

      Correlated updates across windows:
        e_new = rho * e_old + sqrt(1-rho^2) * Std(h) * z,  z~N(0,1)

    Note: randomness only changes the Load parameter, so optimization remains an LP.
    """
    if error_kwargs is None:
        error_kwargs={}
    n_steps = N_T - N_t + 1

    base = data.data()
    storage_inputs = _extract_storage_inputs(base)
    N_s = storage_inputs["N_s"]

    TLMP = np.zeros((N_g, n_steps))
    LLMP = np.zeros((N_g, n_steps))
    P_LAED = np.zeros((N_g, n_steps))
    R_LAED = np.zeros((N_g, n_steps))
    RP_LAED = np.zeros((N_g, n_steps))
    Shed_LAED = np.zeros(n_steps)
    SoC_LAED  = np.zeros((N_s, n_steps)) if N_s > 0 else np.zeros((0, n_steps))
    Pch_LAED  = np.zeros((N_s, n_steps)) if N_s > 0 else np.zeros((0, n_steps))
    Pdis_LAED = np.zeros((N_s, n_steps)) if N_s > 0 else np.zeros((0, n_steps))

    # "True" underlying load (scaled) for each physical time in 1..N_T
    load_true = {t: base["Load"][t] * load_factor for t in base["Load"]}

    gen_init  = {g: base["Gen_init"][g] * load_factor for g in base["Gen_init"]}
    ramp_init = {g: base["Ramp_lim"][g] * ramp_factor for g in base["Ramp_lim"]}

    cap_init  = dict(base["Capacity"])
    cost_init = dict(base["Cost"])
    reserve_single = base["reserve_single"] if isinstance(base["reserve_single"], (int, float)) else base["reserve_single"][None]

    model_laed = laed_opt_model()
    gen_init_current = dict(gen_init)
    soc_init_current = dict(storage_inputs["SoC_init"])

    # Persistent forecast error state per PHYSICAL time index t (1..N_T)
    # This is what creates correlation across rolling windows.
    err_state = {t: 0.0 for t in range(1, N_T + 1)}


    for k in range(n_steps):
        T0 = k + 1

        # At current time, you "observe" load, so forecast error is 0 at lead=0
        err_state[T0] = 0.0

        # Build a windowed forecast load for t = T0..T0+N_t-1
        load_window = {}

        for tt in range(1, N_t + 1):
            t_phys = T0 + tt - 1
            h = tt - 1  # lead time within the window, 0..N_t-1

            if h == 0:
                # perfect knowledge at commit time
                e = 0.0
                err_state[t_phys] = 0.0
            else:
                # Std grows like sqrt(h) -> Var grows like h (linear)
                std_h = sigma_rel * abs(load_true[t_phys]) * np.sqrt(h)

                # Correlated revision vs previous window's forecast error for this same physical time
                e = correlated_error_update(
                    prev_err=err_state[t_phys],
                    std_h=std_h,
                    rho=rho,
                    rng=rng,
                    error_type=error_type,
                    **error_kwargs
                )
                err_state[t_phys] = e

            Lhat = load_true[t_phys] + e
            if clip_nonnegative:
                Lhat = max(0.0, Lhat)

            load_window[tt] = float(Lhat)

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

        results = solver.solve(laed, tee=False, load_solutions=False)
        term = str(results.solver.termination_condition).lower()
        status = str(results.solver.status).lower()
        if term != "optimal" or status != "ok":
            raise RuntimeError(f"LAED infeasible/unknown at window {k+1}: status={status}, term={term}")
        laed.solutions.load_from(results)

        P_laed, Shed_laed, TLMP_T, LLMP_T, _, _, R_laed, Rp_laed = TLMP_calculation(laed, N_g, N_t)

        # Commit t=1 within the window
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

        # Update for next window's initial condition (explicit by generator index, safe for N_g=10 etc.)
        gen_init_current = {g: float(P_laed[g - 1, 0]) for g in range(1, N_g + 1)}
        if storage_inputs["N_s"] > 0:
            soc_init_current = {s: float(pyo.value(laed.SoC[s, 1])) for s in range(1, storage_inputs["N_s"] + 1)}

    return P_LAED, Shed_LAED, TLMP, LLMP, R_LAED, RP_LAED, SoC_LAED, Pch_LAED, Pdis_LAED


def plot_demand_soc_dispatch_shedding(
    *,
    load: dict,
    P_laed: np.ndarray,
    shed_laed: np.ndarray,
    soc_laed: np.ndarray | None = None,
    P_rp: np.ndarray | None = None,
    shed_rp: np.ndarray | None = None,
    soc_rp: np.ndarray | None = None,
    title: str = "Demand, Dispatch, Storage SoC, and Load Shedding",
    show: bool = True,
    out_path: str | None = None,
):
    """
    Plot committed time series for:
      - demand (load)
      - generator dispatch (LAED vs RP)
      - storage state-of-charge (LAED vs RP)
      - load shedding (LAED vs RP)

    Notes:
      - All inputs are assumed to be committed (t=1) time series from rolling simulations.
      - This function aligns series by truncating to a common prefix length.
    """
    if not isinstance(load, dict):
        raise TypeError("load must be a dict indexed by physical time (1..N_T)")

    # Determine common plotting length.
    n_load = len(load)
    n_laed = int(P_laed.shape[1]) if P_laed.size else 0
    n_shed_laed = int(len(shed_laed))
    n = min(n_load, n_laed, n_shed_laed)
    if P_rp is not None and P_rp.size:
        n = min(n, int(P_rp.shape[1]))
    if shed_rp is not None and len(shed_rp) > 0:
        n = min(n, int(len(shed_rp)))

    if n <= 0:
        raise ValueError("Nothing to plot: one or more required series are empty.")

    t = np.arange(1, n + 1)
    load_series = np.array([float(load[k]) for k in range(1, n + 1)], dtype=float)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["C0", "C1", "C2", "C3"])

    # --- Demand + dispatch ---
    ax = axes[0]
    ax.plot(t, load_series, color="k", linewidth=2.0, label="Demand")

    n_g = int(P_laed.shape[0]) if P_laed.size else 0
    for gi in range(n_g):
        c = colors[gi % len(colors)]
        ax.plot(t, P_laed[gi, :n], color=c, linestyle="-", label=f"Gen {gi+1} LAED")
        if P_rp is not None and P_rp.size:
            ax.plot(t, P_rp[gi, :n], color=c, linestyle="--", label=f"Gen {gi+1} RP")

    ax.set_ylabel("Power (MW)")
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=2)

    # --- Storage SoC + load shedding (2 y-axes) ---
    ax = axes[1]
    ax_soc = ax  # left axis
    ax_shed = ax.twinx()  # right axis

    have_soc = soc_laed is not None and soc_laed.size > 0
    have_soc_rp = soc_rp is not None and soc_rp.size > 0
    if have_soc or have_soc_rp:
        n_s = 0
        if have_soc:
            n_s = int(soc_laed.shape[0])
        if have_soc_rp:
            n_s = max(n_s, int(soc_rp.shape[0]))
        for si in range(n_s):
            c = colors[(n_g + si) % len(colors)]
            if have_soc and si < soc_laed.shape[0]:
                ax_soc.plot(t, soc_laed[si, :n], color=c, linestyle="-", linewidth=2.0, label="SoC LAED")
            if have_soc_rp and si < soc_rp.shape[0]:
                ax_soc.plot(t, soc_rp[si, :n], color=c, linestyle="--", linewidth=2.0, label="SoC RP")
    else:
        ax_soc.text(0.01, 0.5, "No storage SoC series", transform=ax_soc.transAxes, va="center")

    ax_shed.plot(t, np.asarray(shed_laed)[:n], color="C3", linestyle="-.", label="Load shed LAED")
    if shed_rp is not None and len(shed_rp) > 0:
        ax_shed.plot(t, np.asarray(shed_rp)[:n], color="C3", linestyle=":", label="Load shed RP")

    ax_soc.set_ylabel("SoC (MWh)")
    ax_shed.set_ylabel("Load shed (MW)")
    ax_soc.set_xlabel("Time (5min Intervals)")
    ax_soc.grid(True, alpha=0.3)

    # Combine legends from both y-axes.
    h1, l1 = ax_soc.get_legend_handles_labels()
    h2, l2 = ax_shed.get_legend_handles_labels()
    ax_soc.legend(h1 + h2, l1 + l2, ncol=2)

    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    if out_path:
        fig.savefig(out_path, dpi=200)
    if show:
        plt.show()
    plt.close(fig)




if __name__=="__main__":
    #system configs
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
    sigma = 0.0 #std of error as percentage of load projection
    rho = 0.95 #correlation between previous prediction and current prediction

    #set the seed
    rng = np.random.default_rng(seed)

    file_path = 'MISO_Projection.json'
    with open(file_path, 'r') as f:
        interpolated_data = json.load(f)

    avg_demand=350
    #avg_demand=2500

    Aug_2032_ori = interpolated_data['2032_Aug']
    load_scale_2032 = avg_demand /(sum(Aug_2032_ori.values())/len(Aug_2032_ori))

    Aug_2032 = {int(key): value*load_scale_2032 for key, value in Aug_2032_ori.items()}

    data.data()["Load"] = Aug_2032
    # The projection is 288 points at 5-minute resolution (24h). Ensure storage SoC dynamics
    # use the correct timestep (hours) instead of whatever is in the .dat file.
    data.data()["delta_t"][None] = 1.0 / 12.0

    # nts = np.linspace(1, 19, 19)
    # laed_sheds = []
    # rp_sheds = []

    # for i in tqdm(range(1,20)):

    #     #define window size
    #     data.data()["N_t"][None] = int(i)

    # #define window size
    data.data()["N_t"][None] = 13
    
    #define policy parameters
    N_g = data.data()['N_g'][None]
    N_t = data.data()['N_t'][None]
    N_t_rp = 2
    N_T = data.data()["N_T"][None]
    N_T = len(Aug_2032_ori)
    cost_init = data.data()['Cost']

    if N_g != 2:
        Load_ini = data.data()['Load'][1]
        # Create an abstract model
        model_ini = AbstractModel()
        model_ini.N_g = Param(within=NonNegativeIntegers) # Number of Generators
        model_ini.G = RangeSet(1, model_ini.N_g)  # Set of Generators
        model_ini.Cost = Param(model_ini.G)
        model_ini.Capacity = Param(model_ini.G)
        model_ini.reserve_single = Param()
        # Define variables
        model_ini.P = Var(model_ini.G,  within=NonNegativeReals)
        model_ini.Reserve = Var(model_ini.G, within=NonNegativeReals)

        # Objective function: Minimize cost
        def objective_rule(model):
            return sum(model.Cost[g] * model.P[g] for g in model.G) 
        model_ini.obj = Objective(rule=objective_rule, sense=minimize)

        # Power balance constraints
        def power_balance_rule(model):
            return sum(model.P[g] for g in model.G) == Load_ini 
        model_ini.power_balance_constraint = Constraint(rule=power_balance_rule)

        # Capacity constraints
        def capacity_rule(model,g):
            return model.P[g] + model.Reserve[g] <= model.Capacity[g]
        model_ini.capacity_constraint = Constraint(model_ini.G, rule=capacity_rule)

        # Reserve constraints, total reserve is reserve_factor of the total load
        def reserve_rule(model):
            return sum(model.Reserve[g] for g in model.G) >= reserve_factor * Load_ini
        model_ini.reserve_constraint = Constraint(rule=reserve_rule)

        # Single Generator Reserve Bid
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
        
    P_ED, Shed_ED, LMP, TLMP, rup_ED, rupp_Ed, rdw_ED, rdwp_Ed, SoC_ED, Pch_ED, Pdis_ED = ED_with_error(
        data_ed, N_g, N_t_rp, N_T, load_factor, ramp_factor, solver, sigma, rho, rng, error_type="gaussian"
    )
    #P_LAED, Shed_LAED, Curt_LAED, TLMP, LLMP, R_LAED, RP_LAED = LAED_No_Errors(data_laed, N_g, N_t, N_T,load_factor, ramp_factor, solver)
    P_LAED, Shed_LAED, TLMP, LLMP, R_LAED, RP_LAED, SoC_LAED, Pch_LAED, Pdis_LAED = LAED_with_error(data_laed, N_g, N_t, N_T, load_factor, ramp_factor, solver, sigma, rho, rng, error_type="gaussian")

    # laed_sheds.append(np.sum(Shed_LAED)/12)
    # rp_sheds.append(np.sum(Shed_ED)/12)

    # #compare results

    # diffs = [laed_sheds[i] - rp_sheds[i] for i in range(len(laed_sheds))]
    # print(diffs)

    # plt.figure()
    # plt.plot(nts, laed_sheds, label="LAED")
    # plt.plot(nts, rp_sheds, label="RP")
    # plt.xlabel('Window Size')
    # plt.ylabel("Load Shedding")
    # plt.xticks(nts.astype(int))
    # plt.legend()
    # plt.show()



    print('Mean Demand:', avg_demand)
    print('Load Shedding in LAED:', np.sum(Shed_LAED)/12)
    print('Load Shedding in ED with only 10-min ramp product:', np.sum(Shed_ED)/12)
    
    plot_demand_soc_dispatch_shedding(
        load=data.data()["Load"],
        P_laed=P_LAED,
        shed_laed=Shed_LAED,
        soc_laed=SoC_LAED,
        P_rp=P_ED,
        shed_rp=Shed_ED,
        soc_rp=SoC_ED,
        title=f"LAED vs RP 1 Hour Look-Ahead with Battery",
        show=True,
        out_path=None,
    )
