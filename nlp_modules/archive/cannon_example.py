
#https://nlopt.readthedocs.io/en/latest/NLopt_Python_Reference/
import nlopt
import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt


class CannonParams():
    def __init__(self):
        pass

def optimize_initial_condition(x):
    """optimize the cannon problem"""
    opt = nlopt.opt(nlopt.nlopt.LD_MMA, 2)
    
    #no lower bounds
    opt.set_min_objective(cost_function)
    
    #m means multiple constraints
    tol = [1e-4, 1e-4]
    opt.add_equality_mconstraint(equality_constraints, tol)
    
    # Set relative tolerance on optimization parameters
    opt.set_xtol_rel(1e-4)
    
    solution = opt.optimize(x)
    
    return solution

def myfunc(x, grad):
    """cost function for cannonball for x and y positions"""
    return np.square(x[0]) + np.square(x[1])

def cost_function(x,grad):
    return np.square(x[0]) + np.square(x[1])

def equality_constraints(result, x, grad):
    """
    For details of the API please refer to:
    https://nlopt.readthedocs.io/en/latest/NLopt_Python_Reference/#:~:text=remove_inequality_constraints()%0Aopt.remove_equality_constraints()-,Vector%2Dvalued%20constraints,-Just%20as%20for
    Note: Please open the link in Chrome
    """
    pos = sim_cannon_dynamics(x)    
    goal_x = 5.0
    goal_y = 0.0
    result[0] = x[0] - goal_x
    result[1] = x[1] - goal_y

def sim_cannon_dynamics(t,S,c):
    """simulate cannon dynamics"""
    g = 9.81
    x,y, vx, vy = S
    
    v = np.sqrt(np.square(vx) + np.square(vy))
    
    ds0 = vx 
    ds1 = vy
    ds2 = -c*v*ds0
    ds3 = -c*v*ds1 - g

    return [ds0, ds1, ds2, ds3]
    
    
if __name__=='__main__':
    plt.close('all')    
    params = {}

    grid_space = 1000
    init_speed = 20 #m/s
    
    init_angle = 50 *np.pi/180
    target_x = 6.0
    target_y = 0
    
    drag_coeff = 0.1
    
    params["v0"] = init_speed
    params["th0"] = init_angle
    params["c"] = drag_coeff

    c = params["c"]     
    x0 = 0
    y0 = 0
    th0 = params["th0"]
    
    dx0 = params["v0"]*np.cos(params["th0"])
    dy0 = params["v0"]*np.sin(params["th0"])
    
    t_start = 0
    t_end = 2.05
    t_span = [t_start,t_end]
    t_eval=np.linspace(t_start, t_end ,1000)

    init_guesses = [0, 0, dx0, dy0]
    solution = integrate.solve_ivp(sim_cannon_dynamics, t_span, init_guesses,
                                   t_eval=t_eval, args=(c,))    

    #plot trajectory
    plt.plot(solution.y[0], solution.y[1])
    plt.show()

    #%% Set up problem
    #set initial guesses from ODE45
    
    #x,y,vx, vy
    x = [solution.y[0][-1], 
                    solution.y[1][-1],
                    solution.y[2][-1],
                    solution.y[3][-1]]
    
    
    opt = nlopt.opt(nlopt.LN_BOBYQA, 4)
    
    # opt.set_lower_bounds([-float('inf'), 0, -float('inf'), -float('inf')]) #can comment this out to default none
    # opt.set_upper_bounds([float('inf'), float('inf'), float('inf'), float('inf')]) #can comment this out to default none
    
    opt.set_min_objective(myfunc)

    tol = [1e-4, 1e-4]
    opt.add_equality_mconstraint(equality_constraints,tol)
    sol = opt.optimize(init_guesses)
    opt.add_equality_mconstraint(lambda x, grad: equality_constraints(x,grad), 1e-4)

    # opt.add_inequality_constraint(lambda x,grad: myconstraint(x,grad,2,0), 1e-8)
    # opt.add_inequality_constraint(lambda x,grad: myconstraint(x,grad,-1,1), 1e-8)
    
    #%%
    opt.set_xtol_rel(1e-4)
    x = opt.optimize([1.234, 5.678])
    minf = opt.last_optimum_value()
    
    print("optimum at ", x[0], x[1])
    print("minimum value = ", minf)
    print("result code = ", opt.last_optimize_result())
    