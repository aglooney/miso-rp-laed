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
    m.Gen_init = Param(m.G)
    m.Ramp_single = Param()
    m.Reserve_single = Param()

    # decision variables
    m.P = Var(m.G, m.T, within=NonNegativeReals)
    m.Reserve = Var(m.G, m.T, within=NonNegativeReals)
    m.Rampup = Var(m.G, m.T, within=NonNegativeReals)
    m.Rampdown = Var(m.G, m.T, within=NonNegativeReals)
    m.Loadshed = Var(m.T, within=NonNegativeReals)
    m.Curtailed = Var(m.T, within=NonNegativeReals)

    m.RU_sys = Var(m.T, within=NonNegativeReals)
    m.RD_sys = Var(m.T, within=NonNegativeReals)

    # objective
    def obj_func(m):
        return (
            sum(m.Cost[g] * m.P[g, t] for g in m.G for t in m.T)
            + cost_load * sum(m.Loadshed[t] for t in m.T)
            + cost_curtailed * sum(m.Curtailed[t] for t in m.T)
        )
    m.obj = Objective(rule=obj_func, sense=minimize)

    # no curtailment option (as you had)
    def curtailment_rule(m, t):
        return m.Curtailed[t] == 0
    m.curtailment_constraint = Constraint(m.T, rule=curtailment_rule)

    # capacity headroom (covers energy + reserve + upward ramp product)
    def capacity_rule(m, g, t):
        return m.P[g, t] + m.Reserve[g, t] + m.Rampup[g, t] <= m.Capacity[g]
    m.capacity_constraint = Constraint(m.G, m.T, rule=capacity_rule)

    # power balance (kept in your exact form; Curtailed is fixed to 0 anyway)
    def power_balance_rule(m, t):
        return sum(m.P[g, t] for g in m.G) == m.Load[t] - m.Loadshed[t] + m.Curtailed[t]
    m.power_balance_constraint = Constraint(m.T, rule=power_balance_rule)

    # ramping of dispatch across time
    def ramp_up_rule(m, g, t):
        if t == 1:
            return m.P[g, 1] - m.Gen_init[g] <= m.Ramp_lim[g]
        return m.P[g, t] - m.P[g, t - 1] <= m.Ramp_lim[g]
    m.ramp_up_constraint = Constraint(m.G, m.T, rule=ramp_up_rule)

    def ramp_down_rule(m, g, t):
        if t == 1:
            return m.Gen_init[g] - m.P[g, 1] <= m.Ramp_lim[g]
        return m.P[g, t - 1] - m.P[g, t] <= m.Ramp_lim[g]
    m.ramp_down_constraint = Constraint(m.G, m.T, rule=ramp_down_rule)

    # per-interval ramp product deliverability bounds
    def rampup_cap_rule(m, g, t):
        return m.Rampup[g, t] <= m.Ramp_lim[g]
    m.rampup_cap_constraint = Constraint(m.G, m.T, rule=rampup_cap_rule)

    def rampdown_cap_rule(m, g, t):
        return m.Rampdown[g, t] <= m.Ramp_lim[g]
    m.rampdown_cap_constraint = Constraint(m.G, m.T, rule=rampdown_cap_rule)

    def rampdown_energy_rule(m, g, t):
        # can't ramp down more than current output
        return m.Rampdown[g, t] <= m.P[g, t]
    m.rampdown_energy_constraint = Constraint(m.G, m.T, rule=rampdown_energy_rule)

    # system ramp requirement definitions: RU_sys[t] >= netload(t+1) - netload(t), RD_sys[t] >= netload(t) - netload(t+1)
    def ru_def_rule(m, t):
        if t == m.T.last():
            return Constraint.Skip
        net_t = (m.Load[t] - m.Loadshed[t] + m.Curtailed[t])
        net_tp1 = (m.Load[t + 1] - m.Loadshed[t + 1] + m.Curtailed[t + 1])
        return m.RU_sys[t] >= (net_tp1 - net_t)
    m.ru_def_constraint = Constraint(m.T, rule=ru_def_rule)

    def rd_def_rule(m, t):
        if t == m.T.last():
            return Constraint.Skip
        net_t = (m.Load[t] - m.Loadshed[t] + m.Curtailed[t])
        net_tp1 = (m.Load[t + 1] - m.Loadshed[t + 1] + m.Curtailed[t + 1])
        return m.RD_sys[t] >= (net_t - net_tp1)
    m.rd_def_constraint = Constraint(m.T, rule=rd_def_rule)

    # system ramp requirements: sum unit ramp products >= RU_sys / RD_sys each interval
    def ru_req_rule(m, t):
        if t == m.T.last():
            return Constraint.Skip
        return sum(m.Rampup[g, t] for g in m.G) >= m.RU_sys[t]
    m.rampup_reserve_constraint = Constraint(m.T, rule=ru_req_rule)  # keep your name, but now indexed

    def rd_req_rule(m, t):
        if t == m.T.last():
            return Constraint.Skip
        return sum(m.Rampdown[g, t] for g in m.G) >= m.RD_sys[t]
    m.rampdw_reserve_constraint = Constraint(m.T, rule=rd_req_rule)  # keep your name, but now indexed

    # reserve requirement at t=1 (as you had)
    def reserve_rule(m):
        return sum(m.Reserve[g, 1] for g in m.G) >= reserve_factor * m.Load[1]
    m.reserve_constraint = Constraint(rule=reserve_rule)

    # single generator reserve cap at t=1 (as you had)
    def reserve_single_rule(m, g):
        return m.Reserve[g, 1] <= m.Reserve_single * m.Capacity[g]
    m.reserve_single_constraint = Constraint(m.G, rule=reserve_single_rule)

    # single generator ramp product caps per-interval (replaces window-end-only cap)
    def rampup_single_rule(m, g, t):
        if t == m.T.last():
            return Constraint.Skip
        return m.Rampup[g, t] <= m.Ramp_single * m.Ramp_lim[g]
    m.rampup_single_constraint = Constraint(m.G, m.T, rule=rampup_single_rule)

    def rampdw_single_rule(m, g, t):
        if t == m.T.last():
            return Constraint.Skip
        return m.Rampdown[g, t] <= m.Ramp_single * m.Ramp_lim[g]
    m.rampdw_single_constraint = Constraint(m.G, m.T, rule=rampdw_single_rule)

    # dual suffix
    m.dual = Suffix(direction=Suffix.IMPORT)

    return m

def LMP_calculation(model):
    t_commit = 1

    P_value = np.array([value(model.P[g, t_commit]) for g in model.G])
    R_value = np.array([value(model.Reserve[g, t_commit]) for g in model.G])
    rup_value = np.array([value(model.Rampup[g, t_commit]) for g in model.G])
    rdw_value = np.array([value(model.Rampdown[g, t_commit]) for g in model.G])

    loadshed_value = value(model.Loadshed[t_commit])
    curtailed_value = value(model.Curtailed[t_commit])

    lam = model.dual.get(model.power_balance_constraint[t_commit], 0.0)

    if model.T.last() >= 2 and (t_commit in model.rampup_reserve_constraint):
        mu_rup = model.dual.get(model.rampup_reserve_constraint[t_commit], 0.0)
    else:
        mu_rup = 0.0
    
    if model.T.last() >= 1 and (t_commit in model.rampup_reserve_constraint):
        mu_rdw = model.dual.get(model.rampdw_reserve_constraint[t_commit], 0.0)
    else:
        mu_rdw = 0.0

    #mu_rup = model.dual.get(model.rampup_reserve_constraint[t_commit], 0.0)
    #mu_rdw = model.dual.get(model.rampdw_reserve_constraint[t_commit], 0.0)
    mu_res = model.dual.get(model.reserve_constraint, 0.0)

    # Your custom “LMP” combination (note: abs() is not standard)
    LMP = abs(lam)
    TLMP = abs(lam)  - abs(mu_rup) + abs(mu_rdw)

    R_price = abs(mu_res)
    rup_price = abs(mu_rup)
    rdw_price = abs(mu_rdw)

    return P_value, loadshed_value, curtailed_value, LMP, TLMP, R_value, R_price, rup_value, rup_price, rdw_value, rdw_price


def ED_no_errors(data, N_g, N_t, N_T, load_factor, ramp_factor, solver):
    n_steps = N_T

    LMP      = np.zeros((N_g, n_steps))
    TLMP     = np.zeros((N_g, n_steps))
    P_LMP    = np.zeros((N_g, n_steps))
    R_ED     = np.zeros((N_g, n_steps))
    RP_ED    = np.zeros((N_g, n_steps))
    rup_ED   = np.zeros((N_g, n_steps))
    rupp_Ed  = np.zeros((N_g, n_steps))
    rdw_ED   = np.zeros((N_g, n_steps))
    rdwp_Ed  = np.zeros((N_g, n_steps))
    Shed_ED  = np.zeros(n_steps)
    Curt_ED  = np.zeros(n_steps)

    base = data.data()

    load_init = {t: base["Load"][t] * load_factor for t in base["Load"]}
    gen_init = {g: base["Gen_init"][g] * load_factor for g in base["Gen_init"]}
    ramp_init = {g: base["Ramp_lim"][g] * ramp_factor for g in base["Ramp_lim"]}

    cap_init  = dict(base["Capacity"])
    cost_init = dict(base["Cost"])

    ramp_single = base["Ramp_single"] if isinstance(base["Ramp_single"], (int, float)) else base["Ramp_single"][None]
    reserve_single = base["Reserve_single"] if isinstance(base["Reserve_single"], (int, float)) else base["Reserve_single"][None]

    rped_model = rped_opt_model()
    gen_init_current = dict(gen_init)

    #print(f"NUMBER OF STEPS: {n_steps}")

    for k in range(n_steps):

        T0 = k + 1

        H = min(N_t, N_T - T0 + 1)

        # windowed load reindexed to 1..N_t
        load_window = {tt: load_init[T0 + tt - 1] for tt in range(1, H + 1)}

        instance_data = {
            None: {
                "N_g": {None: N_g},
                "N_t": {None: H},
                "Cost": cost_init,
                "Capacity": cap_init,
                "Ramp_lim": ramp_init,
                "Gen_init": gen_init_current,
                "Load": load_window,
                "Ramp_single": {None: ramp_single},
                "Reserve_single": {None: reserve_single},
            }
        }

        inst = rped_model.create_instance(data=instance_data)
        results = solver.solve(inst, load_solutions=True, tee=True)

        results = solver.solve(inst, tee=True, load_solutions=True)

        tc = results.solver.termination_condition
        st = results.solver.status

        #if (st != SolverStatus.ok) or (tc not in (TerminationCondition.optimal, TerminationCondition.locallyOptimal)):
        #    raise RuntimeError(f"Solve failed: status={st}, termination={tc}")

        # for v in inst.component_objects(Var, active=True):
        #     print("Variable",v)
        #     for index in v:
        #         if v[index].value is None:
        #             print(f"{v.name}[{index}] = UNINITIALIZED")
        #         else:
        #             print(f"{v.name}[{index}] = {v[index].value}")

        P_ed, Shed_ed, Curt_ed, LMP_ed, TLMP_ed, R_ed, Rp_ed, rup_ed, rupp_ed, rdw_ed, rdwp_ed = LMP_calculation(inst)

        LMP[:, k]     = LMP_ed
        TLMP[:, k]    = TLMP_ed
        P_LMP[:, k]   = P_ed
        R_ED[:, k]    = R_ed
        RP_ED[:, k]   = Rp_ed
        rup_ED[:, k]  = rup_ed
        rupp_Ed[:, k] = rupp_ed
        rdw_ED[:, k]  = rdw_ed
        rdwp_Ed[:, k] = rdwp_ed
        Shed_ED[k]    = Shed_ed
        Curt_ED[k]    = Curt_ed

        # update initial condition for next window
        gen_init_current = {g: float(P_ed[g-1]) for g in range(1, N_g + 1)}

    return P_LMP, Shed_ED, Curt_ED, LMP, TLMP, R_ED, RP_ED, rup_ED, rupp_Ed, rdw_ED, rdwp_Ed


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
    m.Reserve_single = Param()

    # Vars
    m.P = Var(m.G, m.T, within=NonNegativeReals)
    m.Reserve = Var(m.G, m.T, within=NonNegativeReals)
    m.Loadshed = Var(m.T, within=NonNegativeReals)
    m.Curtailed = Var(m.T, within=NonNegativeReals)

    # Objective
    def objective_rule(m):
        return (
            sum(m.Cost[g] * m.P[g, t] for g in m.G for t in m.T)
            + cost_load * sum(m.Loadshed[t] for t in m.T)
            + cost_curtailed * sum(m.Curtailed[t] for t in m.T)
        )
    m.obj = Objective(rule=objective_rule, sense=minimize)

    # No curtailment option (as you had)
    def curtailment_rule(m, t):
        return m.Curtailed[t] == 0
    m.curtailment_constraint = Constraint(m.T, rule=curtailment_rule)

    # Capacity
    def capacity_rule(m, g, t):
        return m.P[g, t] + m.Reserve[g, t] <= m.Capacity[g]
    m.capacity_constraint = Constraint(m.G, m.T, rule=capacity_rule)

    # Power balance
    def power_balance_rule(m, t):
        return sum(m.P[g, t] for g in m.G) == m.Load[t] - m.Loadshed[t] + m.Curtailed[t]
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

    # Reserve requirement each t (as you had)
    def reserve_rule(m, t):
        return sum(m.Reserve[g, t] for g in m.G) >= reserve_factor * m.Load[t]
    m.reserve_constraint = Constraint(m.T, rule=reserve_rule)

    # Single generator reserve cap
    def reserve_single_rule(m, g, t):
        return m.Reserve[g, t] <= m.Reserve_single * m.Capacity[g]
    m.reserve_single_constraint = Constraint(m.G, m.T, rule=reserve_single_rule)

    # Dual suffix
    m.dual = Suffix(direction=Suffix.IMPORT)

    return m

def TLMP_calculation(model, N_g, N_t):
    H = model.T.last()

    P_value = np.array([[value(model.P[g, t]) for t in model.T] for g in model.G], dtype=float)
    R_value = np.array([[value(model.Reserve[g, t]) for t in model.T] for g in model.G], dtype=float)
    loadshed_value = np.array([value(model.Loadshed[t]) for t in model.T], dtype=float)
    curtailed_value = np.array([value(model.Curtailed[t]) for t in model.T], dtype=float)

    la = np.array([model.dual.get(model.power_balance_constraint[t], 0.0) for t in model.T], dtype=float)

    mu_down = np.array([[model.dual.get(model.ramp_down_constraint[g, t], 0.0) for t in model.T] for g in model.G], dtype=float)
    mu_up = np.array([[model.dual.get(model.ramp_up_constraint[g, t], 0.0) for t in model.T] for g in model.G], dtype=float)

    R_price = np.array([model.dual.get(model.reserve_constraint[t], 0.0) for t in model.T], dtype=float)

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

    return P_value, loadshed_value, curtailed_value, TLMP_T, la, mu_down, mu_up, R_value, R_price


def LAED_No_Errors(data, N_g, N_t, N_T, load_factor, ramp_factor, solver):
    n_steps = N_T

    TLMP = np.zeros((N_g, n_steps))
    LLMP = np.zeros((N_g, n_steps))
    P_LAED = np.zeros((N_g, n_steps))
    R_LAED = np.zeros((N_g, n_steps))
    RP_LAED = np.zeros((N_g, n_steps))
    Shed_LAED = np.zeros(n_steps)
    Curt_LAED = np.zeros(n_steps)

    base = data.data()

    # scale base inputs once
    load_init = {t: base["Load"][t] * load_factor for t in base["Load"]}
    gen_init = {g: base["Gen_init"][g] * load_factor for g in base["Gen_init"]}
    ramp_init = {g: base["Ramp_lim"][g] * ramp_factor for g in base["Ramp_lim"]}

    cap_init  = dict(base["Capacity"])
    cost_init = dict(base["Cost"])

    reserve_single = base["Reserve_single"] if isinstance(base["Reserve_single"], (int, float)) else base["Reserve_single"][None]

    # Build model once
    model_laed = laed_opt_model()

    # Rolling initial condition
    gen_init_current = dict(gen_init)

    for k in range(n_steps):
        T0 = k + 1

        H = min(N_t, N_T - T0 + 1)

        # windowed load, reindex to 1..N_t
        load_window = {tt: load_init[T0 + tt - 1] for tt in range(1, H + 1)}

        instance_data = {
            None: {
                "N_g": {None: N_g},
                "N_t": {None: H},
                "N_T": {None: N_T},   # included because model defines it (even if unused)

                "Cost": cost_init,
                "Capacity": cap_init,
                "Ramp_lim": ramp_init,
                "Load": load_window,
                "Gen_init": gen_init_current,

                "Reserve_single": {None: reserve_single},
            }
        }

        laed = model_laed.create_instance(data=instance_data)
        solver.solve(laed, tee=False)

        P_laed, Shed_laed, Curt_laed, TLMP_T, LLMP_T, _, _, R_laed, Rp_laed = TLMP_calculation(laed, N_g, N_t)

        # Commit first period results
        P_LAED[:, k] = P_laed[:, 0]
        R_LAED[:, k] = R_laed[:, 0]

        # TLMP placeholder + LLMP = energy dual at t=1 (replicated per generator like you did)
        TLMP[:, k] = TLMP_T[:, 0]
        LLMP[:, k] = LLMP_T[0] * np.ones(N_g)

        # reserve price at t=1 replicated
        RP_LAED[:, k] = Rp_laed[0] * np.ones(N_g)

        Shed_LAED[k] = Shed_laed[0]
        Curt_LAED[k] = Curt_laed[0]

        # update Gen_init for next window using committed dispatch
        gen_init_current = {g: float(P_laed[g-1, 0]) for g in range(1, N_g + 1)}

    return P_LAED, Shed_LAED, Curt_LAED, TLMP, LLMP, R_LAED, RP_LAED

if __name__=="__main__":
    #system configs
    solver = SolverFactory("gurobi_direct")
    #solver = SolverFactory("glpk")
    solver.options['OutputFlag'] = 0
    load_factor = 1.0
    reserve_factor = 0
    ramp_factor = 0.1
    cost_load = 1e10
    cost_curtailed = 0
    case_name = 'toy_data.dat'
    data = DataPortal()
    data.load(filename=case_name)

    #load historical projection
    # file_path = "MISO_Projection.json"
    # with open(file_path) as f:
    #     interp_data = json.load(f)
    # ref_cap = 325
    # aug_2032_ori = interp_data['2032_Aug']
    # load_scale_2032 = ref_cap/(np.sum(list(aug_2032_ori.values())) / len(aug_2032_ori))
    # aug_2032 = {int(key): value * load_scale_2032 for key, value in aug_2032_ori.items()}
    
    #define policy parameters

    N_g = data.data()['N_g'][None]
    N_t = data.data()['N_t'][None]
    N_T = data.data()["N_T"][None]
    cost_init = data.data()['Cost']

    #N_T = len(aug_2032_ori)
    #data.data()["Load"] = aug_2032
    Gen1_ini = np.min([data.data()['Capacity'][1],data.data()['Load'][1]])
    data.data()['Gen_init'] = {1: Gen1_ini, 2: np.min([data.data()['Capacity'][2], data.data()['Load'][1]- Gen1_ini])}

    data_laed = copy.deepcopy(data)
    data_ed = copy.deepcopy(data)
        
    P_ED, Shed_ED, Curt_ED, LMP, TLMP, R_ED, RP_ED, rup_ED, rupp_Ed, rdw_ED, rdwp_Ed = ED_no_errors(data_ed, N_g, N_t, N_T, load_factor, ramp_factor, solver)
    #P_LAED, Shed_LAED, Curt_LAED, TLMP, LLMP, R_LAED, RP_LAED = LAED_No_Errors(data_laed, N_g, N_t, N_T,load_factor, ramp_factor, solver)
    #compare results


    #print(f"LAED Pricing: {P_LAED}")
    
    # plt.figure()
    times = np.linspace(1,12,12)
    # plt.plot(times, np.array(P_ED)[0], color='blue', label="Generator 1")
    # plt.plot(times, np.array(P_ED)[1], color="orange", label="Generator2")
    # plt.plot(times, list(data.data()["Load"].values()), color='red', label='Load')
    # plt.xlabel("Time Period")
    # plt.ylabel("Power Quantity for Generators")
    # plt.title("Power Generators Values")
    # plt.legend()
    # plt.show()

    plt.figure()
    plt.plot(times, np.array(LMP)[0], label="Generator 1 LMP", color="blue")
    plt.plot(times, np.array(LMP)[1], label="Generator 2 LMP", color="green")
    plt.plot(times, np.array(TLMP)[0], label="Generator 1 TLMP", color="red")
    plt.plot(times, np.array(TLMP)[1], label="Generator 2 TLMP", color="orange")
    assert np.array(LMP)[0].all() == np.array(LMP)[1].all()
    assert np.array(TLMP)[0].all() == np.array(TLMP)[1].all()
    plt.legend()
    plt.show()
