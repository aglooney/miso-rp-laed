from pyomo.environ import *
import numpy as np
import copy
import matplotlib.pyplot as plt
import json


def rped_opt_model():
    #define abstract model
    model = AbstractModel()
    #parameters about data itself
    model.N_g = Param(within=NonNegativeIntegers) #num generators
    model.N_t = Param(within=NonNegativeIntegers) #window size (int)
    model.G = RangeSet(1, model.N_g) #set of generators
    model.T = RangeSet(1, model.N_t) #set of time windows

    #parameters from data
    model.cost = Param(model.G)
    model.capacity = Param(model.G)
    model.ramp_lim = Param(model.G)
    model.load = Param(model.G)
    model.gen_init = Param(model.G)
    model.ramp_single = Param()
    model.reserve_single = Param()

    #decision variables
    model.P = Var(model.G, within=NonNegativeReals)
    model.reserve = Var(model.G, within=NonNegativeReals)
    model.rampup = Var(model.G, within=NonNegativeReals)
    model.rampdown = Var(model.G, within=NonNegativeReals)
    model.loadshed = Var(model.T, within=NonNegativeReals)
    model.curtailed = Var(model.T, within=NonNegativeReals)

    #objective function
    def obj_func(model):
        return sum(model.cost[g] * model.P[g] for g in model.G) + cost_load*np.sum(model.loadshed[t] for t in model.T) + cost_curtailed*np.sum(model.curtailed[t] for t in model.t)
    model.obj = Objective(rule=obj_func, sense=minimize)


    #constraints
    #Option: Without Considering Curtailment
    def curtailment_rule(model, t):
        return model.curtailed[t] == 0
    model.curtailment_constraint = Constraint(model.T, rule=curtailment_rule)

    def capacity_rule(model, g):
        return model.P[g] + model.Reserve[g] + model.rampup[g, model.N_t] <= model.Capacity[g]
    model.capacity_constraint = Constraint(model.G, rule=capacity_rule)
    # Power balance constraints
    def power_balance_rule(model):
        return np.sum(model.P[g] for g in model.G) == model.Load[1] - model.loadshed[1] + model.curtailed[1]
    model.power_balance_constraint = Constraint(rule=power_balance_rule)

    # Ramping constraints for generator
    def ramp_down_rule(model, g):
        return -model.Ramp_lim[g] <= model.P[g] - model.Gen_init[g]
    model.ramp_down_constraint = Constraint(model.G, rule=ramp_down_rule)

    def ramp_up_rule(model, g):
        return model.P[g] - model.Gen_init[g] <= model.Ramp_lim[g]
    model.ramp_up_constraint = Constraint(model.G, rule=ramp_up_rule)

    # Ramping Reserve constraints
    def rampup_reserve_rule(model):
        return np.sum(model.rampup[g, model.N_t] for g in model.G) >= (model.Load[model.N_t] - model.loadshed[model.N_t] + model.curtailed[model.N_t]) - (model.Load[1] - model.loadshed[1] + model.curtailed[1])
    model.rampup_reserve_constraint = Constraint(rule=rampup_reserve_rule) 

    def rampdw_reserve_rule(model):
        return np.sum(model.rampdw[g, model.N_t] for g in model.G) >= (model.Load[1] - model.loadshed[1] + model.curtailed[1]) - (model.Load[model.N_t] - model.loadshed[model.N_t] + model.curtailed[model.N_t])
    model.rampdw_reserve_constraint = Constraint(rule=rampdw_reserve_rule) 

    # Reserve constraints, Ramp is not part of operation reserve
    def reserve_rule(model):
        return np.sum(model.Reserve[g] for g in model.G) >= reserve_factor * model.Load[1]
    model.reserve_constraint = Constraint(rule=reserve_rule)

    # Single Generator Reserve Bid Willingness
    def reserve_single_rule(model, g):
        return model.Reserve[g]  <= model.reserve_single * model.Capacity[g]
    model.reserve_single_constraint = Constraint(model.G, rule=reserve_single_rule)

    # Single Generator Ramp up/dw Bid Willingness 
    def rampup_single_rule(model, g):
        return model.rampup[g, model.N_t] <= (model.N_t-1) * model.ramp_single * model.Ramp_lim[g]
    model.rampup_single_constraint = Constraint(model.G, rule=rampup_single_rule)

    def rampdw_single_rule(model, g):
        return model.rampdw[g, model.N_t] <= (model.N_t-1) * model.ramp_single * model.Ramp_lim[g]
    model.rampdw_single_constraint = Constraint(model.G, rule=rampdw_single_rule)

    # Attach dual suffix
    model.dual = Suffix(direction=Suffix.IMPORT)

    return model

def LMP_calculation(model):
    # Retrieve the results for power output (P) and reserve 
    P_value = np.array([value(model.P[g]) for g in model.G])
    R_value = np.array([value(model.Reserve[g]) for g in model.G])
    rup_value = np.array([value(model.rampup[g, model.N_t]) for g in model.G])
    rdw_value = np.array([value(model.rampdw[g, model.N_t]) for g in model.G])
    loadshed_value = value(model.loadshed[1])
    curtailed_value = value(model.curtailed[1])

    # Retrieve the dual variables (λ) associated with the power balance constraints
    #LMP = np.abs(np.array(model.dual[model.power_balance_constraint]))

    # Our Version
    LMP = np.abs(np.array(model.dual[model.power_balance_constraint])) - np.abs(np.array(model.dual[model.rampup_reserve_constraint])) + np.abs(np.array(model.dual[model.rampdw_reserve_constraint]))

    # Retrieve the dual variables (μ) associated with the reserve constraints
    R_price = np.abs(np.array([model.dual[model.reserve_constraint]]))

    # ramp up price
    rup_price = np.abs(np.array([model.dual[model.rampup_reserve_constraint]]))
    # ramp down price
    rdw_price = np.abs(np.array([model.dual[model.rampdw_reserve_constraint]]))
    
    return P_value, loadshed_value, curtailed_value, LMP, R_value, R_price, rup_value, rup_price, rdw_value, rdw_price



def ED_no_errors(data, N_g, N_t, N_T, load_factor, ramp_factor):
    LMP = np.zeros((N_g, N_T - N_t + 1))
    P_LMP = np.zeros((N_g, N_T - N_t + 1))
    R_ED = np.zeros((N_g, N_T - N_t + 1))
    RP_ED = np.zeros((N_g, N_T - N_t + 1))
    rup_ED = np.zeros((N_g, N_T - N_t + 1))
    rupp_Ed = np.zeros((N_g, N_T - N_t + 1))
    rdw_ED = np.zeros((N_g, N_T - N_t + 1))
    rdwp_Ed = np.zeros((N_g, N_T - N_t + 1))
    Shed_ED = np.zeros(N_T - N_t + 1)
    Curt_ED = np.zeros(N_T - N_t + 1)

    # User defined Load and Reserve Level
    load_data = {key: value * load_factor for key, value in data.data()['Load'].items()}
    gen_init = {key: value * load_factor for key, value in data.data()['Gen_init'].items()}
    ramp_data = {key: value * ramp_factor for key, value in data.data()['Ramp_lim'].items()}
    camp_init = data.data()['Capacity']
    cost_init = data.data()['Cost']

    # Create a copy of the data
    data_rped = copy.deepcopy(data)
    data_rped.data()['Load'] = load_data
    data_rped.data()['Gen_init'] = gen_init
    data_rped.data()['Ramp_lim'] = ramp_data
    data_rped.data()['Capacity'] = camp_init
    data_rped.data()['Cost'] = cost_init

    for T0 in range(1, N_T - N_t + 2):
        # Roll the time window
        subset_load_data = {t-T0+1: load_data[t] for t in range(T0, T0 + N_t)} 
        data_rped.data()['Load'] = subset_load_data
        # Update the initial generation for the first time window
        if T0 > 1:
            for i, key in enumerate(data_rped.data()['Gen_init'].keys()):
                data_rped.data()['Gen_init'][key] = P_LMP[i, T0 - 2]
        else:
            data_rped.data()['Gen_init'] = gen_init 

    # We do not update the ramp limit in the energy market, because if might affect the LMP calculation
    # For example, even if a gen did not bid in the ramp reserve market, it might still be dispatched in the energy market
    # The role of ramp reserve market is to ensure the enough ramping capability in the future              
        
        rped_model = rped_opt_model()
        rped = rped_model.create_instance(data_rped)
        solver.solve(rped, tee=False)
        P_ed, Shed_ed, Curt_ed, LMP_T, R_ed, Rp_ed, rup_ed, rupp_ed, rdw_ed, rdwp_ed = LMP_calculation(rped)
        LMP[:, T0 - 1] = LMP_T*np.ones(N_g)
        P_LMP[:, T0 - 1] = P_ed
        R_ED[:, T0 - 1] = R_ed
        RP_ED[:, T0 - 1] = Rp_ed*np.ones(N_g)
        rup_ED[:, T0 - 1] = rup_ed
        rupp_Ed[:, T0 - 1] = rupp_ed*np.ones(N_g)
        rdw_ED[:, T0 - 1] = rdw_ed
        rdwp_Ed[:, T0 - 1] = rdwp_ed*np.ones(N_g)
        Shed_ED[T0 - 1] = Shed_ed
        Curt_ED[T0 - 1] = Curt_ed

    return P_LMP, Shed_ED, Curt_ED, LMP, R_ED, RP_ED, rup_ED, rupp_Ed, rdw_ED, rdwp_Ed


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

def LAED_No_Errors(data, N_g, N_t, N_T, load_factor, ramp_factor):
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
        model_laed = laed_opt_model()  
        laed = model_laed.create_instance(data_laed)
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
    solver = SolverFactory('gurobi')
    load_factor = 1.0
    reserve_factor = 0
    ramp_factor = 0.1
    cost_load = 1e10
    cost_curtailed = 0
    case_name = 'toy_data.dat'
    data = DataPortal()
    data.load(filename=case_name)

    #load historical projection
    file_path = "MISO_Projection.json"
    with open(file_path) as f:
        interp_data = json.load(f)
    ref_cap = 325
    aug_2032_ori = interp_data['2032_Aug']
    load_scale_2032 = ref_cap/(sum(aug_2032_ori.values) / len(aug_2032_ori))
    aug_2032 = {int(key): value * load_scale_2032 for key, value in aug_2032_ori.items()}
    
    #define policy parameters
    data.data()["N_t"][None] = 13
    N_g = data.data()['N_g'][None]
    N_t = data.data()['N_t'][None]
    cost_init = data.data()['Cost']

    N_T = len(aug_2032_ori)

    data.data()["Load"] = aug_2032
    Gen1_ini = np.min([data.data()['Capacity'][1],data.data()['Load'][1]])
    data.data()['Gen_init'] = {1: Gen1_ini, 2: np.min([data.data()['Capacity'][2], data.data()['Load'][1]- Gen1_ini])}

    data_laed = copy.deepcopy(data)
    data_ed = copy.deepcopy(data)

    P_LAED, Shed_LAED, Curt_LAED, TLMP, LLMP, R_LAED, RP_LAED = LAED_No_Errors(data_laed, N_g, N_t, N_T,load_factor, ramp_factor)
    P_ED, Shed_ED, Curt_ED, LMP, R_ED, RP_ED, rup_ED, rupp_Ed, rdw_ED, rdwp_Ed = ED_no_errors(data_ed, N_g, N_t, N_T, load_factor, ramp_factor)

    #compare results

    print("Mean Demand:", ref_cap)
    print("Load Shedding in LAED:", np.sum(Shed_LAED)/12)
    print(f"Load Shedding in ED with {str(5*data.data()["N_t"][None]-1)} min ramp product: {np.sum(Shed_ED/12)}")