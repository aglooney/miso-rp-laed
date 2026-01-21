import pyomo.environ as pyo
from pyomo.environ import (
    AbstractModel, Param, RangeSet, Var, Constraint, Objective, Suffix, minimize, DataPortal,
    NonNegativeIntegers, NonNegativeReals, value, SolverStatus, TerminationCondition, SolverFactory
)
import numpy as np
import matplotlib.pyplot as plt

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
    m.ramp_single = Param()
    m.reserve_single = Param()

    # decision variables
    m.P = Var(m.G, within=NonNegativeReals)
    m.Rampup = Var(m.G, m.T, within=NonNegativeReals)
    m.Rampdown = Var(m.G, m.T, within=NonNegativeReals)
    m.Loadshed = Var(m.T, within=NonNegativeReals)

    def objective_rule(m):
        return sum(m.Cost[g] * m.P[g] for g in m.G) + cost_load * sum(m.Loadshed[t] for t in m.T)
    m.obj = Objective(rule=objective_rule, sense=minimize)

    def capacity_up_rule(m, g, t):
        return m.P[g] + m.Rampup[g, t] <= m.Capacity[g]
    m.capacity_up_constraint = Constraint(m.G, m.T, rule=capacity_up_rule)

    def capacity_down_rule(m, g, t):
        return m.P[g] - m.Rampdown[g, t] >= 0
    m.capacity_down_constraint = Constraint(m.G, m.T, rule=capacity_down_rule)

    def power_balance_rule(m):
        return sum(m.P[g] for g in m.G) == m.Load[1] - m.Loadshed[1]
    m.power_balance_constraint = Constraint(rule=power_balance_rule)

    def ramp_down_rule(m, g):
        return m.P[g] - m.Gen_prev[g] >= - m.Ramp_lim[g]
    m.ramp_down_constraint = Constraint(m.G, rule=ramp_down_rule)

    def ramp_up_rule(m, g):
        return m.P[g] - m.Gen_prev[g] <= m.Ramp_lim[g]
    m.ramp_up_constraint = Constraint(m.G, rule=ramp_up_rule)

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


def ED_no_errors(data, N_g, N_t, N_T, load_factor, ramp_factor, solver):
    n_steps = N_T

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

    ramp_single = base["ramp_single"] if isinstance(base["ramp_single"], (int, float)) else base["ramp_single"][None]
    reserve_single = base["reserve_single"] if isinstance(base["reserve_single"], (int, float)) else base["reserve_single"][None]

    rped_model = rped_opt_model()
    gen_prev_current = dict(gen_init)

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
                "Gen_prev": gen_prev_current,
                "Load": load_window,
                "ramp_single": {None: ramp_single},
                "reserve_single": {None: reserve_single},
            }
        }

        inst = rped_model.create_instance(data=instance_data)
        results = solver.solve(inst, load_solutions=True, tee=True)

        tN = inst.T.last()
        # print(f"Loadshed: {[value(inst.Loadshed[t]) for t in inst.T]}")
        # print(f"Load: {[value(inst.Load[t]) for t in inst.T]}")
        # print(f"Total Generation: {sum(value(inst.P[g]) for g in inst.G)}")
        # print(f"Ramp Up: {[value(inst.Rampup[g,tN]) for g in inst.G]}")
        # print(f"Ramp Down: {[value(inst.Rampdown[g, tN]) for g in inst.G]}")
        print("Ramp limits used:", {g: value(inst.Ramp_lim[g]) for g in inst.G})
        print("Gen_prev used:", {g: value(inst.Gen_prev[g]) for g in inst.G})
        print("Dispatch P:", {g: value(inst.P[g]) for g in inst.G})
        print("Ramp deltas:", {g: value(inst.P[g]) - value(inst.Gen_prev[g]) for g in inst.G})


        P_ed, Shed_ed, LMP_ed, TLMP_ed, rup_ed, rupp_ed, rdw_ed, rdwp_ed = LMP_calculation(inst)

        LMP[:, k]     = LMP_ed * np.ones(N_g)
        TLMP[:, k]    = TLMP_ed * np.ones(N_g)
        P_LMP[:, k]   = P_ed
        rup_ED[:, k]  = rup_ed
        rupp_Ed[:, k] = rupp_ed * np.ones(N_g)
        rdw_ED[:, k]  = rdw_ed
        rdwp_Ed[:, k] = rdwp_ed * np.ones(N_g)
        Shed_ED[k]    = Shed_ed

        # update initial condition for next window
        gen_prev_current = {g: float(P_ed[g-1]) for g in range(1, N_g + 1)}

    return P_LMP, Shed_ED, LMP, TLMP, rup_ED, rupp_Ed, rdw_ED, rdwp_Ed


if __name__=="__main__":
    #system configs
    solver = SolverFactory("gurobi_direct")
    solver.options['OutputFlag'] = 0
    load_factor = 1.0
    reserve_factor = 0
    ramp_factor = 1.0
    cost_load = 1e10
    cost_curtailed = 0
    case_name = 'toy_data.dat'
    data = DataPortal()
    data.load(filename=case_name)




    #define policy parameters
    N_g = data.data()['N_g'][None]
    N_t = data.data()['N_t'][None]
    N_T = data.data()["N_T"][None]
    
    cost_init = data.data()['Cost']
    Gen1_ini = np.min([data.data()['Capacity'][1],data.data()['Load'][1]])
    data.data()['Gen_init'] = {1: Gen1_ini, 2: np.min([data.data()['Capacity'][2], data.data()['Load'][1]- Gen1_ini])}
        
    P_ED, Shed_ED, LMP, TLMP, rup_ED, rupp_Ed, rdw_ED, rdwp_Ed = ED_no_errors(data, N_g, N_t, N_T, load_factor, ramp_factor, solver)

    times = np.linspace(1,12,12)
    plt.figure()
    # plt.plot(times, np.array(LMP)[0], label="Generator 1 LMP", color="blue")
    # plt.plot(times, np.array(LMP)[1], label="Generator 2 LMP", color="green")
    # plt.plot(times, np.array(TLMP)[0], label="Generator 1 TLMP", color="red")
    # plt.plot(times, np.array(TLMP)[1], label="Generator 2 TLMP", color="orange")
    
    plt.plot(times, Shed_ED, label="Load Shedding RP")
    plt.plot(times, list(data.data()["Load"].values()), label="Load")
    plt.plot(times, P_ED[0], label="Generator 1 Output")
    plt.plot(times, P_ED[1], label='Generator 2 Output')

    #assert np.array(LLMP)[0].all() == np.array(LLMP)[1].all()
    print(f"LMP: {LMP}")
    print(f"TLMP: {TLMP}")
    assert np.array(LMP)[0].all() == np.array(TLMP)[0].all()
    assert np.array(LMP[0]).all() == np.array(LMP[1]).all()
    assert np.array(TLMP[0]).all() == np.array(TLMP[1]).all()
    assert np.array(LMP[1]).all() == np.array(TLMP[1]).all()
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(times, rup_ED[0], label="Gen 1 Ramp Up")
    plt.plot(times, rup_ED[1], label="Gen 2 Ramp Up")
    plt.plot(times, rdw_ED[0], label="Gen 1 Ramp Down")
    plt.plot(times, rdw_ED[1], label="Gen 2 Ramp Down")
    
    plt.legend()
    plt.show()