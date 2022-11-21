#% Import stuff here
import ipopt
import numpy as np

class PendulumParams():
    """_summary_:
    container for pendulum parameters
    """
    def __init__(self, L, m, g, I, theta_des):
        self.L = L
        self.m = m
        self.g = g 
        self.I = I
        self.theta_des = np.deg2rad(theta_des)

class OptimizePendulum():
    def __init__(self, pendulum_params:PendulumParams, 
                 grid_space:int, n_states:int, dt:int):
        
        self.pendulum_params = pendulum_params
        self.grid_space = grid_space
        self.n_states = n_states
        self.dt = dt

    def objective(self,x):
        """"""
        #convert x from long row vector to [n_states x grid_space] 
        states = x[0:].reshape(self.n_states, -1)
        control = x[1][:]
        cost = self.dt * sum(np.square(control))
        
        return cost
        
    def gradient(self,x):
        """"""
        grad = np.zeros()
        iu = GRID_SPACING + np.arange(1,GRID_SPACING+1)
        grad[iu-1] = 2 * self.h * x[iu-1]
        
        return grad
        
        
    def constraints(self,x):
        """"""
        
    def jacobian(self,x):
        """"""


    def intermediate(
            self,
            alg_mod,
            iter_count,
            obj_value,
            inf_pr,
            inf_du,
            mu,
            d_norm,
            regularization_size,
            alpha_du,
            alpha_pr,
            ls_trials
            ):

        #
        # Example for the use of the intermediate callback.
        #
        print("Objective value at iteration #%d is - %g" % (iter_count, obj_value))


if __name__=='__main__':
    
    n_grid = 100
    t_end = 5 
    dt = t_end/n_grid # dt
    time_span = np.linspace(0, t_end, n_grid)
    
    n_dyn_constraints = 2 
    n_task_constraints = 4 
    n_total_constraints = n_grid - n_dyn_constraints + n_task_constraints 
    
    #pendulum parameters
    pendParams = PendulumParams(L=1, m=1, g=9.81, I=1, theta_des=45)        
    
    x_guess = np.linspace(0,1 ,n_grid)
    u_guess = np.linspace(0,0,n_grid)
    
    X0 = np.concatenate((x_guess,u_guess))
    cl = np.zeros(n_grid) 
    cu = np.zeros(n_grid)
    
    nlp = ipopt.problem(
                n=len(X0),
                m=n_total_constraints,
                problem_obj=OptimizePendulum(pendParams),
                cl=cl,
                cu=cu
                )