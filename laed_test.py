import pyomo.environ as pyo
from pyomo.environ import (
    AbstractModel, Param, RangeSet, Var, Constraint, Objective, Suffix, minimize, DataPortal,
    NonNegativeIntegers, NonNegativeReals, value, SolverStatus, TerminationCondition, SolverFactory
)
import numpy as np
import matplotlib.pyplot as plt
import copy
import json


def laed_opt_model():
    # Create an abstract model
    model_laed = AbstractModel()

    # Define sets
    model_laed.N_g = Param(within=NonNegativeIntegers) # Number of Generators
    model_laed.N_t = Param(within=NonNegativeIntegers) # Window Size
    model_laed.N_T = Param(within=NonNegativeIntegers) # Total Time
    model_laed.G = RangeSet(1, model_laed.N_g)  # Set of Generators
    model_laed.T = RangeSet(1, model_laed.N_t)  # Window 

    # Define parameters
    model_laed.Cost = Param(model_laed.G)
    model_laed.Capacity = Param(model_laed.G)
    model_laed.Ramp_lim = Param(model_laed.G)
    model_laed.Load = Param(model_laed.T)
    model_laed.Gen_init = Param(model_laed.G)
    model_laed.reserve_single = Param()


    # Define variables
    model_laed.P = Var(model_laed.G, model_laed.T, within=NonNegativeReals)
    model_laed.Reserve = Var(model_laed.G, model_laed.T, within=NonNegativeReals)
    model_laed.loadshed = Var(model_laed.T, within=NonNegativeReals)
    model_laed.curtailed = Var(model_laed.T, within=NonNegativeReals)

    # Objective function: Minimize cost
    def objective_rule(model):
        return sum(model.Cost[g] * sum(model.P[g, t] for t in model.T) for g in model.G) + cost_load * sum(model.loadshed[t] for t in model.T) + cost_curtailed * sum(model.curtailed[t] for t in model.T)
        #return sum(model.Cost[g] * model.P[g, 1] for g in model.G) + cost_load * model.loadshed[1] + cost_curtailed * model.curtailed[1]
    model_laed.obj = Objective(rule=objective_rule, sense=minimize)

    # Constraints
    # Option: Without Considering Curtailment 
    # (model_ed.curtailed == 0)
    def curtailment_rule(model, t):
        return model.curtailed[t] == 0
    model_laed.curtailment_constraint = Constraint(model_laed.T, rule=curtailment_rule)

    # Capacity constraints
    def capacity_rule(model, g, t):
        return model.P[g, t] + model.Reserve[g, t] <= model.Capacity[g]
    model_laed.capacity_constraint = Constraint(model_laed.G, model_laed.T, rule=capacity_rule)

    # Power balance constraints
    def power_balance_rule(model, t):
        return sum(model.P[g, t] for g in model.G) == model.Load[t] - model.loadshed[t] + model.curtailed[t]
    model_laed.power_balance_constraint = Constraint(model_laed.T, rule=power_balance_rule)

    # Ramping constraints for generator
    def ramp_down_rule(model, g, t):
        if t == min(model.T):
            return -model.Ramp_lim[g]+ model.Gen_init[g] - model.P[g, t] <= 0 
            #return model.P[g, t] >= 0  # No ramping in the first time step
        else:
            return -model.Ramp_lim[g] + model.P[g, t - 1] - model.P[g, t] <= 0 
    model_laed.ramp_down_constraint = Constraint(model_laed.G, model_laed.T, rule=ramp_down_rule)

    def ramp_up_rule(model, g, t):
        if t == min(model.T):
            return model.P[g, t] - model.Gen_init[g] - model.Ramp_lim[g]<= 0
            #return model.P[g, t] >= 0  # No ramping in the first time step
        else:
            return model.P[g, t] - model.P[g, t - 1] - model.Ramp_lim[g] <= 0 
    model_laed.ramp_up_constraint = Constraint(model_laed.G, model_laed.T, rule=ramp_up_rule)

    # Reserve constraints, total reserve is reserve_factor of the total load
    def reserve_rule(model, t):
        return sum(model.Reserve[g, t] for g in model.G) >= reserve_factor * model.Load[t]
    model_laed.reserve_constraint = Constraint(model_laed.T, rule=reserve_rule)

    # Single Generator Reserve Bid
    def reserve_single_rule(model, g, t):
        return model.Reserve[g, t] <= model.reserve_single * model.Capacity[g]
    model_laed.reserve_single_constraint = Constraint(model_laed.G, model_laed.T, rule=reserve_single_rule)

    # Attach dual suffix
    model_laed.dual = Suffix(direction=Suffix.IMPORT)

    return model_laed


# Return both conventional LMP and TLMP
def TLMP_calculation(model, N_g, N_t):
    # Retrieve the results for power output (P) and reserve output (R)
    P_value = np.array([[value(model.P[g, t]) for t in model.T] for g in model.G])
    R_value = np.array([[value(model.Reserve[g, t]) for t in model.T] for g in model.G])
    loadshed_value = np.array([value(model.loadshed[t]) for t in model.T])
    curtailed_value = np.array([value(model.curtailed[t]) for t in model.T])


    # Retrieve the dual variables (λ) associated with the power balance constraints
    # np.abs to gurantee positive value
    la = np.abs(np.array([model.dual[model.power_balance_constraint[t]] for t in model.T]))

    # Retrieve the dual variables (μ) associated with ramping constraints
    # For ramp-down constraints (Note: np.abs to convert Gurobi dual sign)
    mu_down = np.abs(np.array([[model.dual[model.ramp_down_constraint[g, t]] for t in model.T] for g in model.G]))
    #mu_down = 0

    # For ramp-up constraints (Note: np.abs to convert Gurobi dual sign)
    mu_up = np.abs(np.array([[model.dual[model.ramp_up_constraint[g, t]] for t in model.T] for g in model.G]))
    #mu_up =0 

    # Retrieve the dual variables (λ) associated with the reserve constraints
    # np.abs to gurantee positive value
    R_price = np.abs(np.array([model.dual[model.reserve_constraint[t]] for t in model.T]))

    # Initialize TLMP_T matrix
    TLMP_T = np.zeros((N_g, N_t))

    return P_value, loadshed_value, curtailed_value, TLMP_T, la, mu_down, mu_up, R_value, R_price

def LAED_No_Errors(data, N_g, N_t, N_T, load_factor, ramp_factor, solver):
    TLMP = np.zeros((N_g, N_T - N_t + 1))
    LLMP = np.zeros((N_g, N_T - N_t + 1))
    P_LAED = np.zeros((N_g, N_T - N_t + 1))
    R_LAED = np.zeros((N_g, N_T - N_t + 1))
    RP_LAED = np.zeros((N_g, N_T - N_t + 1))
    Shed_LAED = np.zeros(N_T - N_t + 1)
    Curt_LAED = np.zeros(N_T - N_t + 1)

    # User defined Load and Reserve Level
    load_data = {key: value * load_factor for key, value in data.data()['Load'].items()}
    gen_init = {key: value * load_factor for key, value in data.data()['Gen_init'].items()}
    ramp_data = {key: value * ramp_factor for key, value in data.data()['Ramp_lim'].items()}
    camp_init = data.data()['Capacity']
    cost_init = data.data()['Cost']

    # Create a copy of the data
    data_laed = copy.deepcopy(data)
    data_laed.data()['Load'] = load_data
    data_laed.data()['Gen_init'] = gen_init
    data_laed.data()['Ramp_lim'] = ramp_data
    data_laed.data()['Capacity'] = camp_init
    data_laed.data()['Cost'] = cost_init

    model_laed = laed_opt_model()


    for T0 in range(1, N_T - N_t + 2):
        # Roll the time window
        subset_load_data = {t-T0+1: load_data[t] for t in range(T0, T0 + N_t)}
        data_laed.data()['Load'] = subset_load_data
        # Update the initial generation for the first time window
        if T0 > 1:
            for i, key in enumerate(data_laed.data()['Gen_init'].keys()):
                data_laed.data()['Gen_init'][key] = P_LAED[i, T0 - 2]
        else:
            data_laed.data()['Gen_init'] = gen_init  
        laed = model_laed.create_instance(data = data_laed)
        solver.solve(laed, tee=False)
        P_laed, Shed_laed, Curt_laed, TLMP_T, LLMP_T,_,_,R_laed, Rp_laed = TLMP_calculation(laed, N_g, N_t)

        P_LAED[:, T0 - 1] = P_laed[:, 0]
        R_LAED[:, T0 - 1] = R_laed[:, 0]
        TLMP[:, T0 - 1] = TLMP_T[:, 0]
        LLMP[:,T0 - 1] = LLMP_T[0]*np.ones(N_g)
        RP_LAED[:, T0 - 1] = Rp_laed[0]*np.ones(N_g)
        Shed_LAED[T0 - 1] = Shed_laed[0]
        Curt_LAED[T0 - 1] = Curt_laed[0]

    return P_LAED, Shed_LAED, Curt_LAED, TLMP, LLMP, R_LAED, RP_LAED


if __name__=="__main__":
    #system configs
    solver = SolverFactory("gurobi_direct")
    solver.options['OutputFlag'] = 0
    load_factor = 1.0
    reserve_factor = 0
    ramp_factor = 0.1
    cost_load = 1e10
    cost_curtailed = 0
    case_name = 'toy_data.dat'
    data = DataPortal()
    data.load(filename=case_name)

    file_path = 'MISO_Projection.json'
    with open(file_path, 'r') as f:
        interpolated_data = json.load(f)

    ref_cap=325

    Aug_2032_ori = interpolated_data['2032_Aug']
    load_scale_2032 = ref_cap/(sum(Aug_2032_ori.values())/len(Aug_2032_ori))

    Aug_2032 = {int(key): value*load_scale_2032 for key, value in Aug_2032_ori.items()}

    data.data()["Load"] = Aug_2032

    #define window size
    data.data()["N_t"][None] = 13

    #define policy parameters
    N_g = data.data()['N_g'][None]
    N_t = data.data()['N_t'][None]

    N_T = data.data()["N_T"][None]
    N_T = len(Aug_2032_ori)
    cost_init = data.data()['Cost']

    Gen1_ini = np.min([data.data()['Capacity'][1],data.data()['Load'][1]])
    data.data()['Gen_init'] = {1: Gen1_ini, 2: np.min([data.data()['Capacity'][2], data.data()['Load'][1]- Gen1_ini])}

    data_laed = copy.deepcopy(data)
    #P_LAED, Shed_LAED, Curt_LAED, TLMP, LLMP, R_LAED, RP_LAED = LAED_No_Errors(data_laed, N_g, N_t, N_T,load_factor, ramp_factor, solver)
    P_LAED, Shed_LAED, Curt_LAED, TLMP, LLMP, R_LAED, RP_LAED = LAED_No_Errors(data_laed, N_g, N_t, N_T, load_factor, ramp_factor, solver)
    #compare results


    print('Mean Demand:', ref_cap)
    print('Load Shedding in LAED:', np.sum(Shed_LAED)/12)
    #print('Load Shedding in ED with only', str(5*(data.data()['N_t'][None]-1)), '-min ramp product:', np.sum(Shed_ED)/12)
    
    times = np.linspace(1,len(Shed_LAED), len(Shed_LAED))

    plt.figure()
    plt.plot(times, Shed_LAED, label="LAED")
    #plt.plot(times, Shed_ED, label="RP")
    plt.plot(times, list(data.data()["Load"].values())[:len(times)], label="Load")
    plt.plot(times, P_LAED[0], label="Generator 1")
    plt.plot(times, P_LAED[1], label="Generator 2")
    plt.legend()
    plt.show()
