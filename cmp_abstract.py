# cmp_abstract.py
import pyomo.environ as pyo


def build_cmp_abstract_model(allow_shedding: bool = False):
    m = pyo.AbstractModel()

    # -----------------------------
    # Scalars
    # -----------------------------
    m.N_g = pyo.Param(within=pyo.PositiveIntegers)
    m.N_T = pyo.Param(within=pyo.PositiveIntegers)

    # CMP indices: upcoming interval and end of horizon
    m.t_hat = pyo.Param(within=pyo.PositiveIntegers)
    m.T_end = pyo.Param(within=pyo.PositiveIntegers)

    # -----------------------------
    # Sets
    # -----------------------------
    m.G = pyo.RangeSet(1, m.N_g)
    m.T = pyo.RangeSet(1, m.N_T)

    # Future horizon only (no past variables)
    m.T_future = pyo.RangeSet(m.t_hat, m.T_end)

    # -----------------------------
    # Data
    # -----------------------------
    m.Cost     = pyo.Param(m.G)      # $/MWh
    m.Capacity = pyo.Param(m.G)      # MW
    m.Ramp_lim = pyo.Param(m.G)      # MW per interval (make sure consistent with timestep)
    m.Load     = pyo.Param(m.T)      # MW (you will use only t in [t_hat, T_end])

    # Fixed realized past dispatch at time (t_hat - 1)
    # This is p'_k* in the paper, but you only need the last past value for ramp coupling
    m.P_past_star = pyo.Param(m.G)

    # Realized LAD duals for the past-future coupling constraints (μ''*).
    # For your ramp model, the coupling constraints at the boundary are:
    #   (1) P[g,t_hat] - P_past_star[g] <= Ramp_lim[g]     (ramp-up boundary)
    #   (2) P_past_star[g] - P[g,t_hat] <= Ramp_lim[g]     (ramp-down boundary)
    #
    # These duals should be nonnegative if the constraints are written in <= form.
    m.mu_ru_star = pyo.Param(m.G, default=0.0)  # μ_up''* for boundary ramp-up
    m.mu_rd_star = pyo.Param(m.G, default=0.0)  # μ_down''* for boundary ramp-down

    # Optional load shedding (future intervals only). Kept off by default so the
    # base CMP formulation remains unchanged.
    if allow_shedding:
        m.cost_load = pyo.Param(within=pyo.NonNegativeReals, default=0.0)

    # -----------------------------
    # Variables (future only)
    # -----------------------------
    m.P = pyo.Var(m.G, m.T_future, within=pyo.NonNegativeReals)
    if allow_shedding:
        m.Loadshed = pyo.Var(m.T_future, within=pyo.NonNegativeReals)

    # -----------------------------
    # Objective: CMP
    #
    # min  Σ_{g} Σ_{t=t_hat..T_end} Cost[g]*P[g,t]
    #    + Σ_g (mu_ru_star[g] * (A_row*[0;P_future])  + mu_rd_star[g] * (A_row*[0;P_future]))
    #
    # For boundary ramp-up:  P[g,t_hat] - P_past_star[g] <= Ramp_lim[g]
    #   A*[0;P_future] = +P[g,t_hat]
    # For boundary ramp-down: P_past_star[g] - P[g,t_hat] <= Ramp_lim[g]
    #   A*[0;P_future] = -P[g,t_hat]
    #
    # So objective adjustment = Σ_g (mu_ru_star[g]*P[g,t_hat] - mu_rd_star[g]*P[g,t_hat])
    # -----------------------------
    def obj_rule(m):
        th = int(m.t_hat)

        op_cost = sum(m.Cost[g] * m.P[g, t]
                      for g in m.G for t in m.T_future)

        coupling_dual_term = sum((m.mu_ru_star[g] - m.mu_rd_star[g]) * m.P[g, th]
                                 for g in m.G)

        shed_penalty = 0.0
        if allow_shedding:
            shed_penalty = m.cost_load * sum(m.Loadshed[t] for t in m.T_future)

        return op_cost + coupling_dual_term + shed_penalty

    m.Obj = pyo.Objective(rule=obj_rule, sense=pyo.minimize)

    # -----------------------------
    # System constraint (unchanged): power balance for all future t
    # -----------------------------
    def balance_rule(m, t):
        if allow_shedding:
            return sum(m.P[g, t] for g in m.G) == m.Load[t] - m.Loadshed[t]
        return sum(m.P[g, t] for g in m.G) == m.Load[t]
    m.Balance = pyo.Constraint(m.T_future, rule=balance_rule)

    # -----------------------------
    # Capacity constraints for future t
    # -----------------------------
    def cap_hi_rule(m, g, t):
        return m.P[g, t] <= m.Capacity[g]
    m.CapHi = pyo.Constraint(m.G, m.T_future, rule=cap_hi_rule)

    # -----------------------------
    # Intertemporal ramp constraints in the future horizon
    # plus coupling to fixed past dispatch at t_hat
    # -----------------------------
    def ramp_up_rule(m, g, t):
        th = int(m.t_hat)
        if t == th:
            # coupling constraint A''[p_past*; p_future] <= b''
            return m.P[g, th] - m.P_past_star[g] <= m.Ramp_lim[g]
        # future-only constraint A'''[0; p_future] <= b'''
        return m.P[g, t] - m.P[g, t-1] <= m.Ramp_lim[g]
    m.RampUp = pyo.Constraint(m.G, m.T_future, rule=ramp_up_rule)

    def ramp_dn_rule(m, g, t):
        th = int(m.t_hat)
        if t == th:
            return m.P_past_star[g] - m.P[g, th] <= m.Ramp_lim[g]
        return m.P[g, t-1] - m.P[g, t] <= m.Ramp_lim[g]
    m.RampDn = pyo.Constraint(m.G, m.T_future, rule=ramp_dn_rule)

    # Dual suffix (for λ_t etc. if you want to read prices from Balance)
    m.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)

    return m


if __name__ == "__main__":
    import sys
    dat_path = sys.argv[1]

    model = build_cmp_abstract_model()
    inst = model.create_instance(dat_path)

    solver = pyo.SolverFactory("gurobi")
    res = solver.solve(inst, tee=True)

    print("\nFuture balance duals (λ_t):")
    for t in range(int(inst.t_hat), int(inst.T_end) + 1):
        print(t, inst.dual.get(inst.Balance[t], None))
