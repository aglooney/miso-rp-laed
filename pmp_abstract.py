# pmp_abstract.py
import pyomo.environ as pyo


def build_pmp_abstract_model(allow_shedding: bool = False):
    m = pyo.AbstractModel()

    # -----------------------------
    # Scalar params (model sizes / horizon)
    # -----------------------------
    m.N_g = pyo.Param(within=pyo.PositiveIntegers)   # number of generators
    m.N_T = pyo.Param(within=pyo.PositiveIntegers)   # number of time periods (overall horizon length)

    # PMP indices: earliest past, upcoming interval, ending interval (all in {1,...,N_T})
    m.T0    = pyo.Param(within=pyo.PositiveIntegers)
    m.t_hat = pyo.Param(within=pyo.PositiveIntegers)
    m.T_end = pyo.Param(within=pyo.PositiveIntegers)

    # -----------------------------
    # Sets
    # -----------------------------
    m.G = pyo.RangeSet(1, m.N_g)
    m.T = pyo.RangeSet(1, m.N_T)
    m.T_model = pyo.Set(initialize=lambda m: range(int(m.T0), int(m.T_end) + 1))
    m.T_future = pyo.Set(initialize=lambda m: range(int(m.t_hat), int(m.T_end) + 1))

    # -----------------------------
    # Data params (like your .dat style)
    # -----------------------------
    m.Cost     = pyo.Param(m.G)          # linear marginal cost ($/MWh)
    m.Capacity = pyo.Param(m.G)          # max output
    m.Ramp_lim = pyo.Param(m.G)          # symmetric ramp limit (up/down per interval)
    m.Load     = pyo.Param(m.T)          # demand d_t
    m.Gen_init = pyo.Param(m.G)          # initial generation before period 1 (or before T0)

    # Realized prices for past intervals (only used for t in [T0, t_hat-1])
    # You can still provide values for all t to keep the .dat simple.
    m.Lambda_star = pyo.Param(m.T, default=0.0)

    # Optional load shedding (future intervals only). This keeps the base PMP formulation
    # unchanged unless explicitly enabled by the caller.
    if allow_shedding:
        m.cost_load = pyo.Param(within=pyo.NonNegativeReals, default=0.0)

    # -----------------------------
    # Variables: keep past dispatch as vars too
    # -----------------------------
    m.P = pyo.Var(m.G, m.T, within=pyo.Reals)
    if allow_shedding:
        m.Loadshed = pyo.Var(m.T_future, within=pyo.NonNegativeReals)

    # -----------------------------
    # Objective: PMP(T0, t_hat, T_end)
    #   min sum_{g,t in [T0,T_end]} Cost_g * P_{g,t}
    #     + sum_{t=T0..t_hat-1} lambda*_t * (-sum_g P_{g,t} + Load_t)
    # -----------------------------
    def obj_rule(m):
        T0, th, Tend = int(m.T0), int(m.t_hat), int(m.T_end)

        op_cost = sum(m.Cost[g] * m.P[g,t]
                      for g in m.G for t in range(T0, Tend+1))

        penalty = sum(m.Lambda_star[t] * (-sum(m.P[g,t] for g in m.G) + m.Load[t])
                      for t in range(T0, th))

        shed_penalty = 0.0
        if allow_shedding:
            shed_penalty = m.cost_load * sum(m.Loadshed[t] for t in m.T_future)

        return op_cost + penalty + shed_penalty

    m.Obj = pyo.Objective(rule=obj_rule, sense=pyo.minimize)

    # -----------------------------
    # Power balance equality constraints ONLY for future intervals t in [t_hat, T_end]
    # -----------------------------
    def balance_rule(m, t):
        if allow_shedding:
            return sum(m.P[g, t] for g in m.G) == m.Load[t] - m.Loadshed[t]
        return sum(m.P[g,t] for g in m.G) == m.Load[t]

    m.Balance = pyo.Constraint(m.T_future, rule=balance_rule)

    # -----------------------------
    # Capacity constraints for ALL intervals in [T0, T_end]
    # -----------------------------
    def cap_lo_rule(m, g, t):
        return m.P[g,t] >= 0.0
    def cap_hi_rule(m, g, t):
        return m.P[g,t] <= m.Capacity[g]

    m.CapLo = pyo.Constraint(m.G, m.T_model, rule=cap_lo_rule)
    m.CapHi = pyo.Constraint(m.G, m.T_model, rule=cap_hi_rule)

    # -----------------------------
    # Ramping constraints for ALL intervals in [T0, T_end]
    # Use Gen_init for the first modeled period in the horizon
    # -----------------------------
    def ramp_up_rule(m, g, t):
        T0 = int(m.T0)
        if t == T0:
            return m.P[g,t] - m.Gen_init[g] <= m.Ramp_lim[g]
        return m.P[g,t] - m.P[g,t-1] <= m.Ramp_lim[g]

    def ramp_down_rule(m, g, t):
        T0 = int(m.T0)
        if t == T0:
            return m.Gen_init[g] - m.P[g,t] <= m.Ramp_lim[g]
        return m.P[g,t-1] - m.P[g,t] <= m.Ramp_lim[g]

    m.RampUp = pyo.Constraint(m.G, m.T_model, rule=ramp_up_rule)
    m.RampDn = pyo.Constraint(m.G, m.T_model, rule=ramp_down_rule)

    # Duals (useful if you want shadow prices for Balance constraints)
    m.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)

    return m


if __name__ == "__main__":
    # Example solve call:
    # python pmp_abstract.py data/pmp.dat
    import sys
    dat_path = sys.argv[1]

    model = build_pmp_abstract_model()
    inst = model.create_instance(dat_path)

    solver = pyo.SolverFactory("gurobi")
    res = solver.solve(inst, tee=True)

    # Print future interval balance duals (λ_t) if continuous
    print("\nBalance duals (future):")
    for t in range(int(inst.t_hat), int(inst.T_end)+1):
        print(t, inst.dual.get(inst.Balance[t], None))
