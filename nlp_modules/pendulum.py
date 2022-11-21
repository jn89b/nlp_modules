import ipopt
import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt

GRID_SPACING = 30

TARGET_ANGLE = 25 # degrees

L = 1 # length of pendulum (m)
m = 1 # mass of pendulum (kg)
I = 1 # masss moment of inertia (kg m^2)
g = 9.81 # gravity constant (m/s^2)    

class Pendulum():
    def __init__(self, h, n_states, n_unknowns, num_constraints):
        self.h = h
        self.n_states = n_states
        self.n_unknowns = n_unknowns
        self.num_constraints = num_constraints

    def objective(self, x:np.array) -> float:
        """u squared function"""
        states = x[0:].reshape(self.n_states, -1)
        control = states[1][:]
        cost = self.h * sum(np.square(control))
        
        return cost
    
    def gradient(self, x:np.array) -> np.array:
        """compute the gradient of the objective function,
        which is the controls indices of the long row vector"""
        grad = np.zeros(self.n_unknowns)
        iu = GRID_SPACING + np.arange(1,GRID_SPACING+1)
        grad[iu-1] = 2 * self.h * x[iu-1]
        
        return grad    

    def constraints(self, x:np.array):
        """get constraints """
        c = []
        states = x[0:].reshape(self.n_states, -1)
        control = states[1][:]
        
        position = states[0]
        
        """dynamic constraints"""
        for i in range(GRID_SPACING-2):
            x1 = position[i]
            x2 = position[i+1]
            x3 = position[i+2]
            
            u2 = control[i+1]
            xdd = (x3 - 2*x2 + x1)/self.h**2;
            c.append(xdd - ( -m * g * L*np.sin(x2) + u2) / I)
        
        """boundary constraints"""        
        #init position must be 0
        c.append(position[0])
        #init velocity must be 0
        c.append(control[0])
        
        #final position must be at target 
        c.append(position[-1] - (TARGET_ANGLE * np.pi/180))
        #final velocity must be 0
        c.append(control[-1] - control[-2])
        
        
    def jacobian(self, x):
        J = np.zeros((self.num_constraints, self.n_unknowns))

        for i in range(GRID_SPACING-2):
            x2 = x[i+1]
            
            J[i,i] 		= 1/self.h**2;
            J[i,i+1] 	= -2/self.h**2 + m * g * L * np.cos(x2) / I;
            J[i,i+2] 	= 1/self.h**2;
            J[i,GRID_SPACING+i+1] 	= -1/I;
    
		# initial position must be zero:
        J[GRID_SPACING-2,	0] = 1;
        # initial velocity must be zero:
        J[GRID_SPACING-1,1] = 1;
        J[GRID_SPACING-1,0] = -1;
        # final position must be at target angle:
        J[GRID_SPACING, GRID_SPACING] = 1;
        # final velocity must be zero:
        J[GRID_SPACING+1, GRID_SPACING] = 1;    
        J[GRID_SPACING+1, GRID_SPACING-1] = -1;
        print(J)
            
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

        # Example for the use of the intermediate callback.
        #
        print("Objective value at iteration #%d is - %g" % (iter_count, obj_value))


if __name__ == '__main__':
    
    t_end = 50 #seconds
    h = t_end/ GRID_SPACING-1 # time interval between segments
    times = np.linspace(0, t_end, GRID_SPACING-1)
    Ncon = GRID_SPACING - 2 + 4 #N-2 dynamic constraints and 4 task constraints
        
    x = np.linspace(0,1, GRID_SPACING)
    u = np.linspace(0,1, GRID_SPACING)
    
    X0 = np.concatenate((x,u), axis=None)
    cl = np.zeros(Ncon)
    cu = np.zeros(Ncon)
        
    nlp = ipopt.problem(
                n=len(X0),
                m=Ncon,
                problem_obj=Pendulum(h, 2, len(X0), Ncon),
                cl=cl,
                cu=cu
                )
    
    
    x, info = nlp.solve(X0)