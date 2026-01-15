from pyomo.environ import *
import numpy as np
import copy
import matplotlib.pyplot as plt
import json


def ed_rp_opt_model():
    model = AbstractModel()

    model.N_g = Param(within=NonNegativeIntegers)
    model.N_t = Param(within=NonNegativeIntegers)
    model.G = Param(within=NonNegativeIntegers)
    model.T = Param(within=NonNegativeIntegers)

    model.Cost = Param(model.G)
    model.Capacity = Param(model.G)
    model.Ramp_lim = Param(model.G)
    model.Load = Param(model.T)
    model.Gen_init = Param(model.G)
    model.ramp_single = Param()
    model.reserve_single = Param()

    model.P = Var(model.G, within=NonNegativeReals)
    model.Reserve = Var(model.G, within=NonNegativeReals)
    model.rampup = Var(model.G, model_ed.T, within=NonNegativeReals)
    model.rampdw = Var(model.G, model_ed.T, within=NonNegativeReals)
    model.loadshed = Var(model.T,within=NonNegativeReals)
    model.curtailed = Var(model.T,within=NonNegativeReals)


    #objective function
    def objective_function(model):
        return np.sum(model.Cost[g]*model.P[g] for g in model.G) + cost_load*np.sum(model.loadshed[t] for t in model.T) + cost_curtailed*np.sum(model.curtailed[t] for t in model.T)
    model.obj = Objective(rule=objective_function, sense = minimize)


    #constraints
    #option: without considering curtailment
    def curtailment_rule(model):
        return model.curtailed[t] == 0
    model.curtailment_constraint = Constraint(model.T, rule = curtailment_rule)

    # Capacity constraints
    def capacity_rule(model, g):
        return model.P[g] + model.Reserve[g] + model.rampup[g, model.N_t] <= model.Capacity[g]
    model.capacity_constraint = Constraint(model.G, rule=capacity_rule)

    def capacity_rule_dw(model, g):
        return model.P[g] >= model.rampdw[g, model.N_t]
    model.capacitydw_constraint = Constraint(model.G, rule=capacity_rule_dw)

    # Power balance constraints
    def power_balance_rule(model):
        return sum(model.P[g] for g in model.G) == model.Load[1] - model.loadshed[1] + model.curtailed[1]
    model.power_balance_constraint = Constraint(rule=power_balance_rule)

    # Ramping constraints for generator
    def ramp_down_rule(model, g):
        return -model.Ramp_lim[g] <= model.P[g] - model.Gen_init[g]
    model.ramp_down_constraint = Constraint(model.G, rule=ramp_down_rule)

    def ramp_up_rule(model, g):
        return model.P[g] - model.Gen_init[g] <= model.Ramp_lim[g]
    model.ramp_up_constraint = Constraint(model.G, rule=ramp_up_rule)
#############################################################################################
    # Ramping Reserve constraints
    def rampup_reserve_rule(model):
        return sum(model.rampup[g, model.N_t] for g in model.G) >= (model.Load[model.N_t] - model.loadshed[model.N_t] + model.curtailed[model.N_t]) - (model.Load[1] - model.loadshed[1] + model.curtailed[1])
    model.rampup_reserve_constraint = Constraint(rule=rampup_reserve_rule) 

    def rampdw_reserve_rule(model):
        return sum(model.rampdw[g, model.N_t] for g in model.G) >= (model.Load[1] - model.loadshed[1] + model.curtailed[1]) - (model.Load[model.N_t] - model.loadshed[model.N_t] + model.curtailed[model.N_t])
    model.rampdw_reserve_constraint = Constraint(rule=rampdw_reserve_rule) 

    # Reserve constraints, Ramp is not part of operation reserve
    def reserve_rule(model):
        return sum(model.Reserve[g] for g in model.G) >= reserve_factor * model.Load[1]
    model.reserve_constraint = Constraint(rule=reserve_rule)

    # Single Generator Reserve Bid
    def reserve_single_rule(model, g):
        return model.Reserve[g]  <= model.reserve_single * model.Capacity[g]
    model.reserve_single_constraint = Constraint(model.G, rule=reserve_single_rule)

    # Single Generator Ramp up/dw Bid  
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

if __name__ == "__main__":
    solver = SolverFactory('gurobi')
    solver.options['OutputFlag'] = 0
    load_factor = 1.0 # Scale factor for Net Demand
    reserve_factor = 0 # For future research use
    ramp_factor = 0.1 # Scale factor for resource ramping limit
    cost_load = 1e10 # Penalty for load curtailment
    cost_curtailed = 0 # Penalty for curtailed renewable  

    case_name = "10GEN_MASKED.dat"
    data = DataPortal()
    data.load(filename = case_name)

    file_path = "MISO_Projection.json"
    with open(file_path, 'r') as f:
        interp_data = json.load(f)

    ref_cap = 1395 #max=total capacity    
    ref_cap = 1295

    Aug_2032_ori = interp_data['2032_Aug']
    load_scale_2032 = ref_cap / (sum(Aug_2032_ori.values()) / len(Aug_2032_ori))
    aug_2032 = {int(key): value * load_scale_2032 for key, value in Aug_2032_ori.items()}
 
    #laed dispatch policy parameter setting
    # Look ahead window size
    data.data()['N_t'][None] =13

    # Some Constants
    N_g = data.data()['N_g'][None]
    N_t = data.data()['N_t'][None] # Look ahead window size
    cost_init = data.data()['Cost']

    # 24 hours interval
    N_T = len(Aug_2032_ori) # Total time

    