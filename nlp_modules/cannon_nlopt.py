import nlopt
import numpy as np 
from scipy import integrate
import matplotlib.pyplot as plt


"""
Run trajectory optimization for cannon ball
Goal locations are 5, 0


State Space of cannon ball are:
[x, y, v_x, v_y]

Initial inputs are:
[v_mag, theta] 

"""

GOAL_X = 5.0
GOAL_Y = 0.0

def compute_cannon_dynamics(t,solution,c,g):
    """simulate cannon dynamics with state space system
    c = coefficient of drag
    g = gravity constant 
    """
    x,y,vx,vy = solution
    
    v = np.sqrt(np.square(vx) + np.square(vy))
    
    ds0 = vx
    ds1 = vy
    ds2 = -c*v*ds0
    ds3 = -c*v*ds1 - g

    return [ds0, ds1, ds2, ds3]
    
def sim_cannon(sim_parameters):
    """
    compute the cannon dynamics of the system returns the states of
    the cannon as [x,y, vx, and vy] 
    """
    
    v_init = sim_parameters.get('speed')
    theta_init = sim_parameters.get('theta')
    c = sim_parameters.get('c')
    grid_points = sim_parameters.get('grid_points')
    
    g = 9.81 #m/s
    
    x_init = 0 
    y_init = 0
    
    dx0 = v_init*np.cos(theta_init)
    dy0 = v_init*np.sin(theta_init)
    
    if dy0 < 0:
        print("can't shoot cannon through ground sin(theta) > 0 required")
        return
    
    init_guesses = [x_init, y_init, dx0, dy0]

    t_start = 0
    t_end = 2.05
    t_span = [t_start,t_end]
    t_eval=np.linspace(t_start, t_end ,grid_points)
    
    solution = integrate.solve_ivp(compute_cannon_dynamics, 
                                   t_span, 
                                   init_guesses,
                                   t_eval=t_eval,
                                   args=(c,g,))    

    return solution
    
if __name__ == '__main__':
    
    params = {}
    params['speed'] = 20 #m/s
    params['theta'] = np.deg2rad(45)
    params['c'] = 0.1 #drag coefficient
    params['grid_points'] = 100 
    
    solution = sim_cannon(params)
    plt.plot(solution.y[0], solution.y[1])
    plt.show()

    
