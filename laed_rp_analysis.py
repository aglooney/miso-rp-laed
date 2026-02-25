<<<<<<< HEAD
import numpy as np
import copy
import json
import os
import time
import matplotlib.pyplot as plt
from pyomo.environ import (
            AbstractModel, Param, RangeSet, Var, Constraint, Objective, Suffix, minimize, DataPortal,
            NonNegativeIntegers, NonNegativeReals, value, SolverFactory, Set
        )
#from tqdm import tqdm

# Global policy/penalty parameters. Other scripts typically overwrite these on import.
reserve_factor = 0.0
cost_load = 1e10


# def rped_opt_model():
#     m = AbstractModel()

#     # sizes / sets
#     m.N_g = Param(within=NonNegativeIntegers)
#     m.N_t = Param(within=NonNegativeIntegers)
#     m.G = RangeSet(1, m.N_g)
#     m.T = RangeSet(1, m.N_t)

#     # parameters
#     m.Cost = Param(m.G)
#     m.Capacity = Param(m.G)
#     m.Ramp_lim = Param(m.G)
#     m.Load = Param(m.T)
#     m.Gen_prev = Param(m.G)
#     m.ramp_single = Param()
#     m.reserve_single = Param()

#     # decision variables
#     m.P = Var(m.G, within=NonNegativeReals)
#     m.Rampup = Var(m.G, m.T, within=NonNegativeReals)
#     m.Rampdown = Var(m.G, m.T, within=NonNegativeReals)
#     m.Loadshed = Var(m.T, within=NonNegativeReals)

#     def objective_rule(m):
#         return sum(m.Cost[g] * m.P[g] for g in m.G) + cost_load * sum(m.Loadshed[t] for t in m.T)
#     m.obj = Objective(rule=objective_rule, sense=minimize)

#     def capacity_up_rule(m, g, t):
#         return m.P[g] + m.Rampup[g, t] <= m.Capacity[g]
#     m.capacity_up_constraint = Constraint(m.G, m.T, rule=capacity_up_rule)

#     def capacity_down_rule(m, g, t):
#         return m.P[g] - m.Rampdown[g, t] >= 0
#     m.capacity_down_constraint = Constraint(m.G, m.T, rule=capacity_down_rule)

#     def power_balance_rule(m):
#         return sum(m.P[g] for g in m.G) == m.Load[1] - m.Loadshed[1]
#     m.power_balance_constraint = Constraint(rule=power_balance_rule)

#     def ramp_down_rule(m, g):
#         return m.P[g] - m.Gen_prev[g] >= - m.Ramp_lim[g]
#     m.ramp_down_constraint = Constraint(m.G, rule=ramp_down_rule)

#     def ramp_up_rule(m, g):
#         return m.P[g] - m.Gen_prev[g] <= m.Ramp_lim[g]
#     m.ramp_up_constraint = Constraint(m.G, rule=ramp_up_rule)

#     def ru_endpoint_rule(m):
#         tN = m.T.last()
#         return sum(m.Rampup[g, tN] for g in m.G) >= (m.Load[tN] - m.Load[1])
#     m.ru_endpoint_constraint = Constraint(rule=ru_endpoint_rule)

#     def rd_endpoint_rule(m):
#         tN = m.T.last()
#         return sum(m.Rampdown[g, tN] for g in m.G) >= (m.Load[1] - m.Load[tN])
#     m.rd_endpoint_constraint = Constraint(rule=rd_endpoint_rule)


#     def ru_10min_rule(m):
#         if m.T.last() < 2:
#             return Constraint.Skip
#         return sum(m.Rampup[g, 2] for g in m.G) >= (m.Load[2] - m.Load[1])
#     m.ru_10min_constraint = Constraint(rule=ru_10min_rule)

#     def rd_10min_rule(m):
#         if m.T.last() < 2:
#             return Constraint.Skip
#         return sum(m.Rampdown[g, 2] for g in m.G) >= (m.Load[1] - m.Load[2])
#     m.rd_10min_constraint = Constraint(rule=rd_10min_rule)

#     def ru_coupling_rule(m, g):
#         tN = m.T.last()
#         if tN < 2:
#             return Constraint.Skip
#         return m.P[g] + m.Rampup[g, 2] + m.Rampup[g, tN] <= m.Capacity[g]
#     m.ru_coupling_constraint = Constraint(m.G, rule=ru_coupling_rule)

#     def rd_coupling_rule(m, g):
#         tN = m.T.last()
#         if tN < 2:
#             return Constraint.Skip
#         return m.P[g] - m.Rampdown[g, 2] - m.Rampdown[g, tN] >= 0
#     m.rd_coupling_constraint = Constraint(m.G, rule=rd_coupling_rule)





#     # def ru_def_rule(m, t):
#     #     if t == m.T.last():
#     #         return Constraint.Skip
#     #     net_t = (m.Load[t] - m.Loadshed[t])
#     #     net_tp1 = (m.Load[t + 1] - m.Loadshed[t + 1])
#     #     return sum(m.Rampup[g,t] for g in m.G) >= (net_tp1 - net_t)
#     # m.ru_def_constraint = Constraint(m.T, rule=ru_def_rule)

#     # def rd_def_rule(m, t):
#     #     if t == m.T.last():
#     #         return Constraint.Skip
#     #     net_t = (m.Load[t] - m.Loadshed[t])
#     #     net_tp1 = (m.Load[t + 1] - m.Loadshed[t + 1])
#     #     return sum(m.Rampdown[g,t] for g in m.G) >= (net_t - net_tp1)
#     # m.rd_def_constraint = Constraint(m.T, rule=rd_def_rule)
    
#     # def ru_endpoint_rule(m):
#     #     t1 = 1
#     #     tN = m.T.last()
#     #     net_t1 = (m.Load[t1] - m.Loadshed[t1])
#     #     net_tN = (m.Load[tN] - m.Loadshed[tN])
#     #     return sum(m.Rampup[g, tN] for g in m.G) >= (net_tN - net_t1)
#     # m.ru_endpoint_constraint = Constraint(rule=ru_endpoint_rule)

#     # def rd_endpoint_rule(m):
#     #     t1 = 1
#     #     tN = m.T.last()
#     #     net_t1 = (m.Load[t1] - m.Loadshed[t1])
#     #     net_tN = (m.Load[tN] - m.Loadshed[tN])
#     #     return sum(m.Rampdown[g, tN] for g in m.G) >= (net_t1 - net_tN)
#     # m.rd_endpoint_constraint = Constraint(rule=rd_endpoint_rule)

#     m.dual = Suffix(direction = Suffix.IMPORT)

#     return m
=======
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

>>>>>>> origin/main


def rped_opt_model():
    m = AbstractModel()

    # sizes / sets
    m.N_g = Param(within=NonNegativeIntegers)
    m.N_t = Param(within=NonNegativeIntegers)
    m.G = RangeSet(1, m.N_g)
    m.T = RangeSet(1, m.N_t)

    # parameters
    m.Cost = Param(m.G)
    m.Capacity = Param(m.G)
    m.Ramp_lim = Param(m.G)
    m.Load = Param(m.T)
    m.Gen_prev = Param(m.G)
<<<<<<< HEAD

    m.ramp_single = Param()
    m.reserve_single = Param()  # kept for compatibility (unused here)

    # decision variables (NAMES MATCH YOUR LMP_calculation)
=======
    m.ramp_single = Param()
    m.reserve_single = Param()

    # decision variables
>>>>>>> origin/main
    m.P = Var(m.G, within=NonNegativeReals)
    m.Rampup = Var(m.G, m.T, within=NonNegativeReals)
    m.Rampdown = Var(m.G, m.T, within=NonNegativeReals)
    m.Loadshed = Var(m.T, within=NonNegativeReals)

<<<<<<< HEAD
    # objective (NO ramp variables in objective, as you requested)
=======
>>>>>>> origin/main
    def objective_rule(m):
        return sum(m.Cost[g] * m.P[g] for g in m.G) + cost_load * sum(m.Loadshed[t] for t in m.T)
    m.obj = Objective(rule=objective_rule, sense=minimize)

<<<<<<< HEAD
    # single-interval power balance (only t=1)
=======
    def capacity_up_rule(m, g, t):
        return m.P[g] + m.Rampup[g, t] <= m.Capacity[g]
    m.capacity_up_constraint = Constraint(m.G, m.T, rule=capacity_up_rule)

    def capacity_down_rule(m, g, t):
        return m.P[g] - m.Rampdown[g, t] >= 0
    m.capacity_down_constraint = Constraint(m.G, m.T, rule=capacity_down_rule)

>>>>>>> origin/main
    def power_balance_rule(m):
        return sum(m.P[g] for g in m.G) == m.Load[1] - m.Loadshed[1]
    m.power_balance_constraint = Constraint(rule=power_balance_rule)

<<<<<<< HEAD
    # force future shedding to zero (keeps single-interval meaning)
    def future_shed_zero_rule(m, t):
        if t == 1:
            return Constraint.Skip
        return m.Loadshed[t] == 0.0
    m.future_shed_zero_constraint = Constraint(m.T, rule=future_shed_zero_rule)

    # ramping from previous dispatch to current dispatch
    def ramp_down_rule(m, g):
        return -m.Ramp_lim[g] <= m.P[g] - m.Gen_prev[g]
=======
    def ramp_down_rule(m, g):
        return m.P[g] - m.Gen_prev[g] >= - m.Ramp_lim[g]
>>>>>>> origin/main
    m.ramp_down_constraint = Constraint(m.G, rule=ramp_down_rule)

    def ramp_up_rule(m, g):
        return m.P[g] - m.Gen_prev[g] <= m.Ramp_lim[g]
    m.ramp_up_constraint = Constraint(m.G, rule=ramp_up_rule)

<<<<<<< HEAD
    # convenience
    def _tN(m):
        return m.T.last()

    # ---- Ensure unused ramp vars are not uninitialized ----
    # Set ramp at t=1 to 0 (avoids "uninitialized VarData" when Nt=1)
    def ru_zero_t1(m, g):
        return m.Rampup[g, 1] == 0.0
    m.ru_zero_t1_constraint = Constraint(m.G, rule=ru_zero_t1)

    def rd_zero_t1(m, g):
        return m.Rampdown[g, 1] == 0.0
    m.rd_zero_t1_constraint = Constraint(m.G, rule=rd_zero_t1)

    # ---- Capacity constraints for the TWO products (10-min and endpoint) ----
    # Endpoint product always uses tN (if Nt==1, it’s t=1 and we fixed it to 0)
    def capacity_endpoint_rule(m, g):
        tN = _tN(m)
        return m.P[g] + m.Rampup[g, tN] <= m.Capacity[g]
    m.capacity_endpoint_constraint = Constraint(m.G, rule=capacity_endpoint_rule)

    def capacitydw_endpoint_rule(m, g):
        tN = _tN(m)
        return m.P[g] >= m.Rampdown[g, tN]
    m.capacitydw_endpoint_constraint = Constraint(m.G, rule=capacitydw_endpoint_rule)

    # 10-minute product uses t=2 (only if available)
    def capacity_10min_rule(m, g):
        if m.T.last() < 2:
            return Constraint.Skip
        return m.P[g] + m.Rampup[g, 2] <= m.Capacity[g]
    m.capacity_10min_constraint = Constraint(m.G, rule=capacity_10min_rule)

    def capacitydw_10min_rule(m, g):
        if m.T.last() < 2:
            return Constraint.Skip
        return m.P[g] >= m.Rampdown[g, 2]
    m.capacitydw_10min_constraint = Constraint(m.G, rule=capacitydw_10min_rule)

    # ---- Ramp requirement constraints (NAMES MATCH YOUR LMP_calculation) ----
    # Endpoint requirement (this is what your code calls "ru_endpoint_constraint")
    def ru_endpoint_rule(m):
        tN = _tN(m)
        if tN < 2:
            # No meaningful endpoint ramp requirement if window is 1 step
            return Constraint.Skip
        return sum(m.Rampup[g, tN] for g in m.G) >= (m.Load[tN] - m.Load[1])
    m.ru_endpoint_constraint = Constraint(rule=ru_endpoint_rule)

    def rd_endpoint_rule(m):
        tN = _tN(m)
        if tN < 2:
            return Constraint.Skip
        return sum(m.Rampdown[g, tN] for g in m.G) >= (m.Load[1] - m.Load[tN])
    m.rd_endpoint_constraint = Constraint(rule=rd_endpoint_rule)

    # 10-minute requirement (kept, but named separately)
    def ru_10min_rule(m):
        if m.T.last() < 2:
            return Constraint.Skip
        return sum(m.Rampup[g, 2] for g in m.G) >= (m.Load[2] - m.Load[1])
    m.ru_10min_constraint = Constraint(rule=ru_10min_rule)

    def rd_10min_rule(m):
        if m.T.last() < 2:
            return Constraint.Skip
        return sum(m.Rampdown[g, 2] for g in m.G) >= (m.Load[1] - m.Load[2])
    m.rd_10min_constraint = Constraint(rule=rd_10min_rule)

    # ---- Single-generator ramp bid willingness (exactly like your original) ----
    def rampup_single_endpoint_rule(m, g):
        tN = _tN(m)
        if tN < 2:
            return Constraint.Skip
        return m.Rampup[g, tN] <= (tN - 1) * m.ramp_single * m.Ramp_lim[g]
    m.rampup_single_endpoint_constraint = Constraint(m.G, rule=rampup_single_endpoint_rule)

    def rampdw_single_endpoint_rule(m, g):
        tN = _tN(m)
        if tN < 2:
            return Constraint.Skip
        return m.Rampdown[g, tN] <= (tN - 1) * m.ramp_single * m.Ramp_lim[g]
    m.rampdw_single_endpoint_constraint = Constraint(m.G, rule=rampdw_single_endpoint_rule)

    def rampup_single_10min_rule(m, g):
        if m.T.last() < 2:
            return Constraint.Skip
        return m.Rampup[g, 2] <= (2 - 1) * m.ramp_single * m.Ramp_lim[g]
    m.rampup_single_10min_constraint = Constraint(m.G, rule=rampup_single_10min_rule)

    def rampdw_single_10min_rule(m, g):
        if m.T.last() < 2:
            return Constraint.Skip
        return m.Rampdown[g, 2] <= (2 - 1) * m.ramp_single * m.Ramp_lim[g]
    m.rampdw_single_10min_constraint = Constraint(m.G, rule=rampdw_single_10min_rule)

    # duals
    m.dual = Suffix(direction=Suffix.IMPORT)

    return m




=======
    def rampup_window_rule(m, g, t):
        return m.Rampup[g, t] <= (t-1) * m.ramp_single * m.Ramp_lim[g]
    m.rampup_window_constraint = Constraint(m.G, m.T, rule=rampup_window_rule)

    def rampdown_window_rule(m, g, t):
        return m.Rampdown[g, t] <= (t-1) * m.ramp_single * m.Ramp_lim[g]
    m.rampdown_window_constraint = Constraint(m.G, m.T, rule = rampdown_window_rule)

    # def ru_def_rule(m, t):
    #     if t == m.T.last():
    #         return Constraint.Skip
    #     net_t = (m.Load[t] - m.Loadshed[t])
    #     net_tp1 = (m.Load[t + 1] - m.Loadshed[t + 1])
    #     return sum(m.Rampup[g,t] for g in m.G) >= (net_tp1 - net_t)
    # m.ru_def_constraint = Constraint(m.T, rule=ru_def_rule)

    # def rd_def_rule(m, t):
    #     if t == m.T.last():
    #         return Constraint.Skip
    #     net_t = (m.Load[t] - m.Loadshed[t])
    #     net_tp1 = (m.Load[t + 1] - m.Loadshed[t + 1])
    #     return sum(m.Rampdown[g,t] for g in m.G) >= (net_t - net_tp1)
    # m.rd_def_constraint = Constraint(m.T, rule=rd_def_rule)
    
    def ru_endpoint_rule(m):
        t1 = 1
        tN = m.T.last()
        net_t1 = (m.Load[t1] - m.Loadshed[t1])
        net_tN = (m.Load[tN] - m.Loadshed[tN])
        return sum(m.Rampup[g, tN] for g in m.G) >= (net_tN - net_t1)
    m.ru_endpoint_constraint = Constraint(rule=ru_endpoint_rule)

    def rd_endpoint_rule(m):
        t1 = 1
        tN = m.T.last()
        net_t1 = (m.Load[t1] - m.Loadshed[t1])
        net_tN = (m.Load[tN] - m.Loadshed[tN])
        return sum(m.Rampdown[g, tN] for g in m.G) >= (net_t1 - net_tN)
    m.rd_endpoint_constraint = Constraint(rule=rd_endpoint_rule)

    m.dual = Suffix(direction = Suffix.IMPORT)

    return m

>>>>>>> origin/main
def LMP_calculation(model):
    t_commit = 1
    tN = model.T.last()

    P_value = np.array([value(model.P[g]) for g in model.G], dtype=float)
    
    rup_value = np.array([value(model.Rampup[g, tN]) for g in model.G], dtype=float)
    rdw_value = np.array([value(model.Rampdown[g, tN]) for g in model.G], dtype=float)

    loadshed_value = value(model.Loadshed[t_commit])

<<<<<<< HEAD
    lam = float(model.dual.get(model.power_balance_constraint, 0.0))

    if tN < 2:
        # No ramp products defined for a 1-step window
        rup_value = np.zeros(len(list(model.G)), dtype=float)
        rdw_value = np.zeros(len(list(model.G)), dtype=float)
        mu_rup = 0.0
        mu_rdw = 0.0
    else:
        rup_value = np.array([value(model.Rampup[g, tN]) for g in model.G], dtype=float)
        rdw_value = np.array([value(model.Rampdown[g, tN]) for g in model.G], dtype=float)
        # Endpoint ramp requirements are >= constraints. For minimization LPs, duals should be >= 0.
        mu_rup = float(model.dual.get(model.ru_endpoint_constraint, 0.0))
        mu_rdw = float(model.dual.get(model.rd_endpoint_constraint, 0.0))

=======
    lam = model.dual.get(model.power_balance_constraint, 0.0)

    mu_rup = float(model.dual.get(model.ru_endpoint_constraint, 0.0)) if tN >= 2 else 0.0
    mu_rdw = float(model.dual.get(model.rd_endpoint_constraint, 0.0)) if tN >= 2 else 0.0
>>>>>>> origin/main

    #mu_rup = model.dual.get(model.rampup_reserve_constraint[t_commit], 0.0)
    #mu_rdw = model.dual.get(model.rampdw_reserve_constraint[t_commit], 0.0)

<<<<<<< HEAD
    # Use abs() to normalize solver sign conventions to positive price magnitudes.
    LMP = abs(lam)
    TLMP = abs(lam) - abs(mu_rup) + abs(mu_rdw)
=======
    # Your custom “LMP” combination (note: abs() is not standard)
    LMP = abs(lam)
    TLMP = abs(lam)  - abs(mu_rup) + abs(mu_rdw)
>>>>>>> origin/main
    rup_price = abs(mu_rup)
    rdw_price = abs(mu_rdw)

    return P_value, loadshed_value, LMP, TLMP, rup_value, rup_price, rdw_value, rdw_price


def ED_no_errors(data, N_g, N_t, N_T, load_factor, ramp_factor, solver):
    n_steps = N_T - N_t + 1

    LMP      = np.zeros((N_g, n_steps))
    TLMP     = np.zeros((N_g, n_steps))
    P_LMP    = np.zeros((N_g, n_steps))
    rup_ED   = np.zeros((N_g, n_steps))
    rupp_Ed  = np.zeros((N_g, n_steps))
    rdw_ED   = np.zeros((N_g, n_steps))
    rdwp_Ed  = np.zeros((N_g, n_steps))
    Shed_ED  = np.zeros(n_steps)

    base = data.data()

    load_init = {t: base["Load"][t] * load_factor for t in base["Load"]}
    gen_init = {g: base["Gen_init"][g] * load_factor for g in base["Gen_init"]}
    ramp_init = {g: base["Ramp_lim"][g] * ramp_factor for g in base["Ramp_lim"]}

    cap_init  = dict(base["Capacity"])
    cost_init = dict(base["Cost"])

<<<<<<< HEAD
    # Enforce 10-minute ramp product fraction (override any dataset value)
    #ramp_single = RAMP_FRACTION_10MIN
=======
>>>>>>> origin/main
    ramp_single = base["ramp_single"] if isinstance(base["ramp_single"], (int, float)) else base["ramp_single"][None]
    reserve_single = base["reserve_single"] if isinstance(base["reserve_single"], (int, float)) else base["reserve_single"][None]

    rped_model = rped_opt_model()
    gen_prev_current = dict(gen_init)

    for k in range(n_steps):

        T0 = k + 1
<<<<<<< HEAD
=======

>>>>>>> origin/main
        H = N_t

        # windowed load reindexed to 1..N_t
        load_window = {tt: load_init[T0 + tt - 1] for tt in range(1, H + 1)}

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
            }
        }

        rped = rped_model.create_instance(data=instance_data)
        results = solver.solve(rped, tee=False)
<<<<<<< HEAD
        term = str(results.solver.termination_condition).lower()
        status = str(results.solver.status).lower()
        if term != 'optimal' and status != 'ok':
            raise RuntimeError(f"RPED infeasible/unknown at window {k+1}: status={status}, term={term}")
=======
        
>>>>>>> origin/main

        # print(f"Loadshed: {[value(inst.Loadshed[t]) for t in inst.T]}")
        # print(f"Load: {[value(inst.Load[t]) for t in inst.T]}")
        # print(f"Total Generation: {sum(value(inst.P[g]) for g in inst.G)}")
        # print(f"Ramp Up: {[value(inst.Rampup[g,tN]) for g in inst.G]}")
        # print(f"Ramp Down: {[value(inst.Rampdown[g, tN]) for g in inst.G]}")
        # print("Ramp limits used:", {g: value(rped.Ramp_lim[g]) for g in rped.G})
        # print("Gen_prev used:", {g: value(rped.Gen_prev[g]) for g in rped.G})
        # print("Dispatch P:", {g: value(rped.P[g]) for g in rped.G})
        # print("Ramp deltas:", {g: value(rped.P[g]) - value(rped.Gen_prev[g]) for g in rped.G})


        P_ed, Shed_ed, LMP_ed, TLMP_ed, rup_ed, rupp_ed, rdw_ed, rdwp_ed = LMP_calculation(rped)

        LMP[:, k]     = LMP_ed
        TLMP[:, k]    = TLMP_ed
        P_LMP[:, k]   = P_ed
        rup_ED[:, k]  = rup_ed
        rupp_Ed[:, k] = rupp_ed
        rdw_ED[:, k]  = rdw_ed
        rdwp_Ed[:, k] = rdwp_ed
        Shed_ED[k]    = Shed_ed

        # update initial condition for next window
        gen_prev_current = {g: float(P_ed[g-1]) for g in range(1, N_g + 1)}

    return P_LMP, Shed_ED, LMP, TLMP, rup_ED, rupp_Ed, rdw_ED, rdwp_Ed


def laed_opt_model():
    m = AbstractModel()

    # Sets / sizes
    m.N_g = Param(within=NonNegativeIntegers)
    m.N_t = Param(within=NonNegativeIntegers)

    # You had N_T in the model but never used it. Keeping it is fine (not adding anything).
    m.N_T = Param(within=NonNegativeIntegers)

    m.G = RangeSet(1, m.N_g)
    m.T = RangeSet(1, m.N_t)

    # Params
    m.Cost = Param(m.G)
    m.Capacity = Param(m.G)
    m.Ramp_lim = Param(m.G)
    m.Load = Param(m.T)
    m.Gen_init = Param(m.G)
    m.reserve_single = Param()

    # Vars
    m.P = Var(m.G, m.T, within=NonNegativeReals)
    m.Loadshed = Var(m.T, within=NonNegativeReals)
    m.Reserve = Var(m.G, m.T, within=NonNegativeReals)

    # Objective
    def objective_rule(m):
        return sum(sum(m.Cost[g] * m.P[g, t] for g in m.G) 
                   + cost_load * m.Loadshed[t] for t in m.T)
    m.obj = Objective(rule=objective_rule, sense=minimize)

    # Capacity
    def capacity_rule(m, g, t):
        return m.P[g, t] + m.Reserve[g,t] <= m.Capacity[g]
    m.capacity_constraint = Constraint(m.G, m.T, rule=capacity_rule)

    # Power balance
    def power_balance_rule(m, t):
        return sum(m.P[g, t] for g in m.G) == m.Load[t] - m.Loadshed[t]
    m.power_balance_constraint = Constraint(m.T, rule=power_balance_rule)

    # Ramping (fixed to properly link time)
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

    # Dual suffix
    m.dual = Suffix(direction=Suffix.IMPORT)

    return m

def TLMP_calculation(model, N_g, N_t):
<<<<<<< HEAD
    # Retrieve the results for power output (P) and reserve output (R)
    P_value = np.array([[value(model.P[g, t]) for t in model.T] for g in model.G])
    R_value = np.array([[value(model.Reserve[g, t]) for t in model.T] for g in model.G])
    loadshed_value = np.array([value(model.Loadshed[t]) for t in model.T])


    # Power balance duals (lambda): equality constraint, so dual sign is solver-dependent.
    # In our models, this dual is already the marginal cost of load (positive under Gurobi),
    # so we do not take abs().
    LMP = np.array([float(model.dual.get(model.power_balance_constraint[t], 0.0)) for t in model.T], dtype=float)

    # Ramping duals: these are (<=) constraints in a minimization LP, so Gurobi reports them as <= 0 when binding.
    # Convert to nonnegative economic multipliers without using abs().
    mu_down = -np.array(
        [[float(model.dual.get(model.ramp_down_constraint[g, t], 0.0)) for t in model.T] for g in model.G],
        dtype=float,
    )
    mu_up = -np.array(
        [[float(model.dual.get(model.ramp_up_constraint[g, t], 0.0)) for t in model.T] for g in model.G],
        dtype=float,
    )

    # Reserve duals: reserve_constraint is (>=), so the economic multiplier is already >= 0 (typically).
    R_price = np.array([float(model.dual.get(model.reserve_constraint[t], 0.0)) for t in model.T], dtype=float)

    # Clean up numerical noise so tiny dual values don't create TLMP jitter.
    eps = 1e-9
    LMP[np.abs(LMP) < eps] = 0.0
    mu_down[np.abs(mu_down) < eps] = 0.0
    mu_up[np.abs(mu_up) < eps] = 0.0
    R_price[np.abs(R_price) < eps] = 0.0

    # Initialize TLMP_T matrix
    TLMP_T = np.zeros((N_g, N_t))

    # Generator-specific TLMP.
    # User convention: use current minus prior ramping multipliers:
    #   TLMP_g,t = LMP_t + (Δμ_g,t - Δμ_g,t-1),  where Δμ := μ_up - μ_down
    deta = mu_up - mu_down
    for s in range(N_t):
        if s == 0:
            deta_prev = 0.0
        else:
            deta_prev = deta[:, s - 1]
        TLMP_T[:, s] = LMP[s] + (deta[:, s] - deta_prev)
    
    # Monitor the Active Reserve Constraint
    #print('Dual of Reserve',deta)
    return P_value, loadshed_value, TLMP_T, LMP, mu_down, mu_up, R_value, R_price


def plot_prices_over_time(times, rp_lmp, laed_lmp, laed_tlmp, out_path=None, show=True):
    """
    times: 1D array of committed time indices
    rp_lmp, laed_lmp: 1D arrays (system prices)
    laed_tlmp: 2D array shape (N_g, len(times)) (generator-specific)
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    ax1.plot(times, rp_lmp, label="RP LMP")
    ax1.plot(times, laed_lmp, label="LAED LMP")
    ax1.set_ylabel("Price ($/MWh)")
    ax1.grid(True, alpha=0.25)
    ax1.legend()

    # TLMP: 2 lines (one per generator), as requested.
    ax2.plot(times, laed_tlmp[0, :], label="LAED TLMP Gen 1")
    ax2.plot(times, laed_tlmp[1, :], label="LAED TLMP Gen 2")
    ax2.set_ylabel("Price ($/MWh)")
    ax2.set_xlabel("Committed time index")
    ax2.grid(True, alpha=0.25)
    ax2.legend()

    fig.suptitle("LAED vs RP Prices Over Time")
    fig.tight_layout()

    if out_path is not None:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        fig.savefig(out_path, dpi=200)
    if show:
        plt.show()
    return fig

# def TLMP_calculation(model, N_g, N_t):
#     H = model.T.last()

#     P_value = np.array([[value(model.P[g, t]) for t in model.T] for g in model.G], dtype=float)
#     R_value = np.array([[value(model.Reserve[g,t]) for t in model.T] for g in model.G], dtype=float)
#     loadshed_value = np.array([value(model.Loadshed[t]) for t in model.T], dtype=float)

#     LMP = np.array([model.dual.get(model.power_balance_constraint[t], 0.0) for t in model.T], dtype=float)

#     # Pyomo reports duals based on internal canonical form. For minimization with
#     # "<=" constraints, economically meaningful multipliers are typically -dual.
#     mu_down_raw = np.array(
#         [[model.dual.get(model.ramp_down_constraint[g, t], 0.0) for t in model.T] for g in model.G],
#         dtype=float,
#     )
#     mu_up_raw = np.array(
#         [[model.dual.get(model.ramp_up_constraint[g, t], 0.0) for t in model.T] for g in model.G],
#     #     dtype=float,
    # )
    # mu_down = -mu_down_raw
    # mu_up = -mu_up_raw

    # R_price = np.abs(np.array([model.dual.get(model.reserve_constraint[t], 0.0) for t in model.T], dtype=float))

    # TLMP_T = np.zeros((N_g, H), dtype=float)

    # for gi, _ in enumerate(model.G):
    #     for t in range(1, H + 1):
    #         t_idx = t - 1
    #         delta_mu_t = mu_up[gi, t_idx] - mu_down[gi, t_idx]
    #         # User-specified convention: use current minus prior ramp multipliers.
    #         # (mu_up_t - mu_down_t) - (mu_up_{t-1} - mu_down_{t-1})
    #         if t == 1:
    #             delta_mu_prev = 0.0
    #         else:
    #             delta_mu_prev = mu_up[gi, t_idx - 1] - mu_down[gi, t_idx - 1]
    #         TLMP_T[gi, t_idx] = LMP[t_idx] + (delta_mu_t - delta_mu_prev)

    # return P_value, loadshed_value, TLMP_T, LMP, mu_down, mu_up, R_value, R_price


# ----------------------------------------------------------------------
# PMP problem (deterministic pricing, uses prior lambda_hat for t < T_hat)
# ----------------------------------------------------------------------
def pmp_opt_model():
    """Deterministic PMP priced as in (11)-(15), specialized to our ED structure."""
    m = AbstractModel()

    # sizes / sets
    m.N_g = Param(within=NonNegativeIntegers)
    m.N_t = Param(within=NonNegativeIntegers)
    m.T_hat = Param(within=NonNegativeIntegers)  # balance enforced for t >= T_hat
    m.G = RangeSet(1, m.N_g)
    m.T = RangeSet(1, m.N_t)

    # parameters
    m.Cost = Param(m.G)
    m.Load = Param(m.T)
    m.Capacity = Param(m.G)
    m.Ramp_lim = Param(m.G)
    m.Gen_init = Param(m.G)
    m.lambda_hat = Param(m.T, default=0.0)  # λ*_t applied in objective for t < T_hat

    # variables
    m.P = Var(m.G, m.T, within=NonNegativeReals)

    # objective
    def obj_rule(m):
        base_cost = sum(m.Cost[g] * m.P[g, t] for g in m.G for t in m.T)
        lag_term = sum(m.lambda_hat[t] * (-sum(m.P[g, t] for g in m.G) + m.Load[t])
                       for t in m.T if t < m.T_hat)
        return base_cost + lag_term
    m.obj = Objective(rule=obj_rule, sense=minimize)

    # balance enforced only from T_hat onward
    def balance_rule(m, t):
        if t < m.T_hat:
            return Constraint.Skip
        return sum(m.P[g, t] for g in m.G) == m.Load[t]
    m.balance = Constraint(m.T, rule=balance_rule)

    # ramp constraints (same structure as LAED)
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

    # capacity bounds per period
    def capacity_rule(m, g, t):
        return m.P[g, t] <= m.Capacity[g]
    m.capacity_constraint = Constraint(m.G, m.T, rule=capacity_rule)

    # duals
    m.dual = Suffix(direction=Suffix.IMPORT_EXPORT)
    return m


def solve_pmp_from_dataportal(data, T_hat, lambda_hat_vec, solver, current_only=False, T0=1):
    """
    Create and solve a PMP instance using the same data layout as LAED.
    lambda_hat_vec: dict {t: value} or list/np array length N_t for prior λ*_t.
    current_only: if True, shrink horizon to a single binding interval (at T_hat=T0).
    T0: earliest past interval (matches PMP(T0, T_hat, T)).
    Returns dispatch P (array), balance duals λ, ramp duals μ_up/μ_down.
    """
    base = data.data()
    N_g = base["N_g"][None]
    N_t_full = base["N_t"][None]
    # Align effective horizon to match LAED length unless current_only requested
    N_t_eff = 1 if current_only else N_t_full

    # build lambda_hat dict over [T0, T]
    lam_hat = {}
    for idx, t in enumerate(range(T0, T0 + N_t_eff)):
        lam_hat[t] = float(lambda_hat_vec[idx] if isinstance(lambda_hat_vec, (list, tuple, np.ndarray))
                           else lambda_hat_vec.get(t, 0.0))

    # slice load for effective horizon
    load_slice = {t: base["Load"][t] for t in range(T0, T0 + N_t_eff)}
    inst_data = {
        None: {
            "N_g": {None: N_g},
            "N_t": {None: N_t_eff},
            "T_hat": {None: int(T_hat)},
            "Cost": dict(base["Cost"]),
            "Load": load_slice,
            "Capacity": dict(base["Capacity"]),
            "Ramp_lim": dict(base["Ramp_lim"]),
            "Gen_init": dict(base["Gen_init"]),
            "lambda_hat": lam_hat,
        }
    }

    model = pmp_opt_model()
    inst = model.create_instance(data=inst_data)
    results = solver.solve(inst, tee=False, load_solutions=True)
    inst.solutions.load_from(results)

    P_val = np.array([[value(inst.P[g, t]) for t in inst.T] for g in inst.G], dtype=float)
    lam = np.array([inst.dual.get(inst.balance[t], 0.0) if t >= T_hat else lam_hat[t] for t in inst.T], dtype=float)
    mu_up = np.array([[inst.dual.get(inst.ramp_up_constraint[g, t], 0.0) for t in inst.T] for g in inst.G], dtype=float)
    mu_down = np.array([[inst.dual.get(inst.ramp_down_constraint[g, t], 0.0) for t in inst.T] for g in inst.G], dtype=float)
    return P_val, lam, mu_up, mu_down, inst
=======
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
>>>>>>> origin/main

def LAED_No_Errors(data, N_g, N_t, N_T, load_factor, ramp_factor, solver):
    n_steps = N_T - N_t + 1

    TLMP = np.zeros((N_g, n_steps))
    LLMP = np.zeros((N_g, n_steps))
    P_LAED = np.zeros((N_g, n_steps))
    R_LAED = np.zeros((N_g, n_steps))
    RP_LAED = np.zeros((N_g, n_steps))
    Shed_LAED = np.zeros(n_steps)
    #Curt_LAED = np.zeros(n_steps)

    base = data.data()

    load_init = {t: base["Load"][t] * load_factor for t in base["Load"]}
    gen_init  = {g: base["Gen_init"][g] * load_factor for g in base["Gen_init"]}
    ramp_init = {g: base["Ramp_lim"][g] * ramp_factor for g in base["Ramp_lim"]}

    cap_init  = dict(base["Capacity"])
    cost_init = dict(base["Cost"])
    reserve_single = base["reserve_single"] if isinstance(base["reserve_single"], (int, float)) else base["reserve_single"][None]

    
    model_laed = laed_opt_model()
    gen_init_current = dict(gen_init)

    for k in range(n_steps):
<<<<<<< HEAD
        if k == 0:
            TLMP_all = np.zeros((N_g, n_steps, N_t))
=======
>>>>>>> origin/main
        T0 = k + 1

        load_window = {tt: load_init[T0 + tt - 1] for tt in range(1, N_t + 1)}

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
            }
        }

        laed = model_laed.create_instance(data=instance_data)
<<<<<<< HEAD
=======

>>>>>>> origin/main
        results = solver.solve(laed, tee=False)


        # print("\n=== Dual summary by time ===")
        # for t in laed.T:
<<<<<<< HEAD
        # #     lam = laed.dual.get(laed.power_balance_constraint[t], None) if t in laed.power_balance_constraint else None
        # #     lam_econ = -lam if lam is not None else None

        # #     #res = laed.dual.get(laed.reserve_constraint[t], None) if t in laed.reserve_constraint else None

        # #     print(f"\nt = {t}")
        # #     print(f"  LLMP raw     = {lam}")
        # #     print(f"  LLMP econ    = {lam_econ}")
        # #     #print(f"  Reserve dual = {res}")
=======
        #     lam = laed.dual.get(laed.power_balance_constraint[t], None) if t in laed.power_balance_constraint else None
        #     lam_econ = -lam if lam is not None else None

        #     #res = laed.dual.get(laed.reserve_constraint[t], None) if t in laed.reserve_constraint else None

        #     print(f"\nt = {t}")
        #     print(f"  LLMP raw     = {lam}")
        #     print(f"  LLMP econ    = {lam_econ}")
        #     #print(f"  Reserve dual = {res}")
>>>>>>> origin/main

        #     for g in laed.G:
        #         mu_u = laed.dual.get(laed.ramp_up_constraint[g, t], None) if (g, t) in laed.ramp_up_constraint else None
        #         mu_d = laed.dual.get(laed.ramp_down_constraint[g, t], None) if (g, t) in laed.ramp_down_constraint else None
<<<<<<< HEAD
        #         print(f"t={t} \t  g={g}: mu_up={mu_u}, mu_down={mu_d}")
=======
        #         print(f"    g={g}: mu_up={mu_u}, mu_down={mu_d}")
>>>>>>> origin/main


        # TLMP_calculation should infer H from laed.T.last()
        #P_laed, Shed_laed, Curt_laed, TLMP_T, LLMP_T, _, _, R_laed, Rp_laed = TLMP_calculation(laed, N_g, N_t)
        P_laed, Shed_laed, TLMP_T, LLMP_T, _, _, R_laed, Rp_laed = TLMP_calculation(laed, N_g, N_t)
        # Commit t=1 (tau=0)
<<<<<<< HEAD
        TLMP_all[:, k, :] = TLMP_T
=======
>>>>>>> origin/main
        P_LAED[:, k] = P_laed[:, 0]
        R_LAED[:, k] = R_laed[:, 0]
        TLMP[:, k]   = TLMP_T[:, 0]
        LLMP[:, k]   = float(LLMP_T[0]) * np.ones(N_g)
        RP_LAED[:, k]= float(Rp_laed[0]) * np.ones(N_g)
        Shed_LAED[k] = Shed_laed[0]
        #Curt_LAED[k] = Curt_laed[0]

        gen_init_current = {g: float(P_laed[g - 1, 0]) for g in range(1, N_g + 1)}

    #return P_LAED, Shed_LAED, Curt_LAED, TLMP, LLMP, R_LAED, RP_LAED
    return P_LAED, Shed_LAED, TLMP, LLMP, R_LAED, RP_LAED



if __name__=="__main__":
<<<<<<< HEAD
    print("Script started. Loading Pyomo...", flush=True)
    print("Pyomo loaded.", flush=True)

=======
>>>>>>> origin/main
    #system configs
    solver = SolverFactory("gurobi_direct")
    solver.options['OutputFlag'] = 0
    load_factor = 1.0
    reserve_factor = 0
    ramp_factor = 0.1
    cost_load = 1e10
<<<<<<< HEAD
    case_name = 'toy_data.dat'
=======
    cost_curtailed = 0
    case_name = '10GEN_MASKED.dat'
>>>>>>> origin/main
    data = DataPortal()
    data.load(filename=case_name)

    file_path = 'MISO_Projection.json'
    with open(file_path, 'r') as f:
        interpolated_data = json.load(f)

<<<<<<< HEAD
    ref_cap=500
=======
    ref_cap=2561
>>>>>>> origin/main

    Aug_2032_ori = interpolated_data['2032_Aug']
    load_scale_2032 = ref_cap/(sum(Aug_2032_ori.values())/len(Aug_2032_ori))

    Aug_2032 = {int(key): value*load_scale_2032 for key, value in Aug_2032_ori.items()}

    data.data()["Load"] = Aug_2032

<<<<<<< HEAD
    nts = np.linspace(1, 41, 40)
    laed_sheds = []
    rp_sheds = []
    laed_times_s = []
    rp_times_s = []

    for i in range(1,41):

        #     #define window size
        data.data()["N_t"][None] = int(i)

        # # #define window size
        #data.data()["N_t"][None] = 13
=======
    nts = np.linspace(1, 20, 20)
    laed_sheds = []
    rp_sheds = []

    for i in tqdm(range(1,21)):

        #define window size
        data.data()["N_t"][None] = int(i)

        # #define window size
        # data.data()["N_t"][None] = 12
        
>>>>>>> origin/main
        #define policy parameters
        N_g = data.data()['N_g'][None]
        N_t = data.data()['N_t'][None]
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
<<<<<<< HEAD

        t0 = time.perf_counter()
        P_ED, Shed_ED, LMP_ED, TLMP_ED, rup_ED, rupp_Ed, rdw_ED, rdwp_Ed = ED_no_errors(
            data_ed, N_g, N_t, N_T, load_factor, ramp_factor, solver
        )
        rp_times_s.append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        P_LAED, Shed_LAED, TLMP_LAED, LLMP_LAED, R_LAED, RP_LAED = LAED_No_Errors(
            data_laed, N_g, N_t, N_T, load_factor, ramp_factor, solver
        )
        laed_times_s.append(time.perf_counter() - t0)
=======
            
        P_ED, Shed_ED, LMP, TLMP, rup_ED, rupp_Ed, rdw_ED, rdwp_Ed = ED_no_errors(data_ed, N_g, N_t, N_T, load_factor, ramp_factor, solver)
        #P_LAED, Shed_LAED, Curt_LAED, TLMP, LLMP, R_LAED, RP_LAED = LAED_No_Errors(data_laed, N_g, N_t, N_T,load_factor, ramp_factor, solver)
        P_LAED, Shed_LAED, TLMP, LLMP, R_LAED, RP_LAED = LAED_No_Errors(data_laed, N_g, N_t, N_T, load_factor, ramp_factor, solver)
>>>>>>> origin/main

        laed_sheds.append(np.sum(Shed_LAED)/12)
        rp_sheds.append(np.sum(Shed_ED)/12)

    #compare results

<<<<<<< HEAD
    diffs = [laed_sheds[i] - rp_sheds[i] for i in range(40)]
    print(diffs)

    fig, ax_shed = plt.subplots(figsize=(9.5, 5.0))
    ax_time = ax_shed.twinx()

    ax_shed.plot(nts, laed_sheds, label="LAED Load Shed", linewidth=2.0)
    ax_shed.plot(
        nts,
        rp_sheds,
        label="ED+RP Load Shed (10-min + W RP)",
        linewidth=2.0,
    )

    ax_time.plot(nts, laed_times_s, label="LAED Compute Time", linestyle="--", linewidth=1.8)
    ax_time.plot(nts, rp_times_s, label="ED+RP Compute Time", linestyle="--", linewidth=1.8)

    ax_shed.set_xticks(list(range(1, 42, 5)))
    ax_shed.set_xlabel("Window Size (number of 5-min intervals)")
    ax_shed.set_ylabel("Load Shed (MW)")
    ax_time.set_ylabel("Computation time (s)")

    handles1, labels1 = ax_shed.get_legend_handles_labels()
    handles2, labels2 = ax_time.get_legend_handles_labels()
    ax_shed.legend(handles1 + handles2, labels1 + labels2, loc="center left")

    ax_shed.grid(True, alpha=0.25)
    fig.suptitle("LAED and RP Comparison: Load Shedding and Computation Time vs Window Length")
    fig.tight_layout()
=======
    diffs = [laed_sheds[i] - rp_sheds[i] for i in range(20)]
    print(diffs)

    plt.figure()
    plt.scatter(nts, laed_sheds, label="LAED")
    plt.scatter(nts, rp_sheds, label="RP")
    plt.xlabel('Window Size')
    plt.ylabel("Load Shedding")
    plt.legend()
>>>>>>> origin/main
    plt.show()



<<<<<<< HEAD
    print('Mean Demand:', ref_cap)
    print('Load Shedding in LAED:', np.sum(Shed_LAED)/12)
    print('Load Shedding in ED with only', str(5*(data.data()['N_t'][None]-1)), '-min ramp product:', np.sum(Shed_ED)/12)
    
    times_ED = np.linspace(1,len(Shed_ED), len(Shed_ED))
    times_LAED = np.linspace(1, len(Shed_LAED), len(Shed_LAED))

    # Prices over committed time (TLMP is generator-specific, so plot both generators)
    times = np.arange(1, len(Shed_ED) + 1)
    plot_prices_over_time(
        times,
        rp_lmp=LMP_ED[0, :],
        laed_lmp=LLMP_LAED[0, :],
        laed_tlmp=TLMP_LAED,
        out_path="prices_vs_time.png",
        show=True,
    )


    # n_steps = N_T - N_t + 1
    # cons_std = np.full((N_g, n_steps), np.nan)
    # for t in range(1, n_steps + 1):
    #     preds = []
    #     for T0 in range(max(1, t - N_t + 1), min(t, n_steps) + 1):
    #         k2 = T0 - 1
    #         tau = t - T0
    #         preds.append(TLMP_all[:, k2, tau])
    #     if len(preds) >= 2:
    #         P = np.stack(preds, axis=1)   # (N_g, n_preds)
    #         cons_std[:, t-1] = np.std(P, axis=1)
    # times = np.arange(1, cons_std.shape[1] + 1)

    # plt.figure(figsize=(9,4))
    # for g in range(cons_std.shape[0]):
    #     plt.plot(times, cons_std[g], label=f"Gen {g+1} consistency std")
    # plt.xlabel("Time")
    # plt.ylabel("Std of forecasted TLMP for same time")
    # plt.title("Price Consistency Across Rolling Windows")
    # plt.legend()
    # plt.tight_layout()
    # plt.show()


    # print(np.array_equal(LMP_ED[0, :], LMP_ED[1,:]))
    # print(np.array_equal(LLMP_LAED[0, :], LLMP_LAED[1,:]))
    # print(np.array_equal(TLMP_ED[0, :] , TLMP_ED[1, :]))
    # print(np.array_equal(TLMP_LAED[0, :],  TLMP_LAED[1, :]))
    # print(np.array_equal(LLMP_LAED, LMP_ED))
    # print(np.array_equal(LLMP_LAED[0, :], TLMP_LAED[0, :]))
    # plt.figure()
    # plt.plot(times_ED, LMP_ED[0, :], label="RP LMP")
    # #plt.plot(times_ED, LMP_ED[1, :], label="RP LMP Gen 2")
    # #plt.plot(times_ED, TLMP_ED[1, :], label='RP TLMP Gen 2')
    # plt.plot(times_LAED, LLMP_LAED[0, :], label="LAED LMP")
    # plt.plot(times_LAED, TLMP_LAED[0, :], label='LAED TLMP Gen 1')
    # plt.plot(times_LAED, TLMP_LAED[1, :], label='LAED TLMP Gen 2')
    # plt.xlabel("Time")
    # plt.ylabel("Energy Price ($\MWh)")
    # plt.legend()
    # plt.show()

    # plt.figure()
    # plt.plot(times_LAED, list(data.data()["Load"].values())[:len(times_LAED)], label="Load")
    # plt.plot(times_LAED, Shed_LAED, label="LAED Load Shedding")
    # plt.plot(times_ED, Shed_ED, label="RP Load Shedding")
    # # plt.plot(times_ED, P_ED[0], label="Gen 1 RP")
    # # plt.plot(times_ED, P_ED[1], label="Gen 2 RP")
    # # plt.plot(times_LAED, P_LAED[0], label="Gen 1 LAED")
    # # plt.plot(times_LAED, P_LAED[1], label="Gen 2 LAED")
    # plt.xlabel("Time")
    # plt.ylabel("Power (MW)")
    # plt.legend()
    # plt.title("Comparison of Load Shedding under RP (10-min + 60-min) and LAED")
    # plt.show()


    # plt.figure()
    # plt.plot(times_LAED, list(data.data()["Load"].values())[:len(times_LAED)], label="Load")
    # plt.plot(times_ED, P_ED[0], label="Gen 1 RP")
    # plt.plot(times_ED, P_ED[1], label="Gen 2 RP")
    # plt.plot(times_LAED, P_LAED[0], label="Gen 1 LAED")
    # plt.plot(times_LAED, P_LAED[1], label="Gen 2 LAED")
    # plt.xlabel("Time")
    # plt.ylabel("Power (MW)")
    # plt.title("Comparison of Generator Values under RP (10-min + 60-min) and LAED")
    # plt.legend()
    # plt.show()
=======
    # print('Mean Demand:', ref_cap)
    # print('Load Shedding in LAED:', np.sum(Shed_LAED)/12)
    # print('Load Shedding in ED with only', str(5*(data.data()['N_t'][None]-1)), '-min ramp product:', np.sum(Shed_ED)/12)
    
    # times = np.linspace(1,len(Shed_ED), len(Shed_ED))

    # plt.figure()
    # plt.plot(times, Shed_LAED, label="LAED Load Shedding")
    # plt.plot(times, Shed_ED, label="RP Load Shedding")
    # plt.plot(times, list(data.data()["Load"].values())[:len(times)], label="Load")
    # # plt.plot(times, P_ED[0], label="Gen 1 RP")
    # # plt.plot(times, P_ED[1], label="Gen 2 RP")
    # # plt.plot(times, P_LAED[0], label="Gen 1 LAED")
    # # plt.plot(times, P_LAED[1], label="Gen 2 LAED")
    # plt.legend()
    # plt.show()
>>>>>>> origin/main
