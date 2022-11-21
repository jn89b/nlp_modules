# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 13:04:00 2022

@author: jnguy
"""

#% Import stuff here
import nlopt
import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt


"""
Block move example: states are the following

x = [x v]
x_dot = [v , u]

"""

#% Classes 

grid_space = 20
n_states = 3
N_BOUND_CONSTRAINTS = 4 # we have 2 states both with a lower and upper bound 

def cost_function(x, grad):
    
    states = x[1:].reshape(n_states, -1)
    control = states[2]

    #u^2 cost function
    cost = sum(np.square(control))
    
    #min time function
    # cost = x[0] #
    
    return cost

def inequality_constraints(result, x, grad):
    """inequality constraints for block move"""
    x_position = x[0]
    result[0] = x_position[0][:] - 5 

def equality_constraints(result, x, grad):
    """equality constraints for block move"""
    #Reshape the long row vector into [n_states x grid_space] vector
    t_final = x[0]
    states = x[1:].reshape(n_states, -1)
    x_position = states[0][:]
    velocity = states[1][:]
    control = states[2][:]

    # result[0] = t_final
    result[1] = x_position[0] #initial x position constraint
    result[2] = x_position[-1] - 1 #final position constraint 
    
    result[3] = velocity[0] # control intial constraint
    result[4] = velocity[-1] # final control constraint
    
    next_iter = N_BOUND_CONSTRAINTS + 1
        
    dt = t_final/grid_space

    for k in range(len(x_position)-1):

        x_curr = np.array([x_position[k], velocity[k]])
        x_new = np.array([x_position[k+1], velocity[k+1]])    
        
        #xdots
        x_dot_curr = np.array([velocity[k], control[k]])
        x_dot_new = np.array([velocity[k+1], control[k+1]])
        
        #trapezoid rule
        x_end = x_curr + dt * ((x_dot_curr + x_dot_new) /2)
        
        x_diff = x_new - x_end
        
        for x_d in x_diff:
            result[next_iter] = x_d
            next_iter = next_iter + 1
    
#% Main 
if __name__=='__main__':
    
    # parameters 
    x_0 = 0 #position m 
    v_0 = 0 # velocity m/s
    u_0 = 0 # initial force function
    
    x_final = 1 
    v_final = 0
    u_final = 0
    
    t_0 = 0 #seconds
    t_final = 1 
    
    #states of system
    x_init = [x_0, v_0]
    
#%% Collocation Method: Go from continous to discret time domain using trape rule
    
    t_eval = np.linspace(t_0, t_final, grid_space)
    x_eval = np.linspace(x_0, 1, grid_space)
    v_eval = np.linspace(v_0, 0, grid_space)#v_0 * np.ones(x_eval.shape)
    u_eval = np.linspace(u_0, 0, grid_space)
    
    x = [x_eval, v_eval, u_eval]
    
    t_span = [t_0, t_final]
    
    dt = np.diff(t_eval)[0]
    
    x_low_bounds = -float('inf') * np.ones(x_eval.shape)
    v_low_bounds = -float('inf') * np.ones(x_eval.shape)
    u_low_bounds = -10 * np.ones(x_eval.shape)

    x_upp_bounds = float('inf') * np.ones(x_eval.shape)
    v_upp_bounds = float('inf') * np.ones(x_eval.shape)
    u_upp_bounds = 1 * np.ones(x_eval.shape)

    low_bounds = np.concatenate((0, x_low_bounds, v_low_bounds, u_low_bounds), axis=None)
    upp_bounds = np.concatenate((float('inf'), x_upp_bounds, v_upp_bounds, u_upp_bounds), axis=None)

#%% Optimize trajectory
    
    constraint_length = n_states * grid_space + 1
    opt = nlopt.opt(nlopt.LN_COBYLA, constraint_length)
    opt.set_maxeval(2000)
    opt.set_xtol_abs(1E-4)    
    
    opt.set_min_objective(cost_function)
    # opt.set_lower_bounds(low_bounds)
    # opt.set_upper_bounds(upp_bounds)   
    
    tol = 1e-2 * np.ones(constraint_length)
    
    opt.add_equality_mconstraint(equality_constraints, tol)

    #need to combine states into single row vector
    x_concat = np.concatenate((t_final, x_eval, v_eval, u_eval), axis=None)
    sol = opt.optimize(x_concat)

    
    #%%
    plt.close('all')
    t_final = sol[0]
    dt = t_final / grid_space 
    states = sol[1:].reshape(n_states, -1)
    
    t_sim = np.arange(0, t_final, dt)
    plt.plot(t_sim, states[0], label='position')
    plt.plot(t_sim, states[1], label='velocity')
    # plt.plot(t_eval, states[2], label='acceleration')

    plt.legend()
    
    
    
    