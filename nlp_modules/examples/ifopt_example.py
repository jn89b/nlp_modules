# -*- coding: utf-8 -*-
"""
Created on Fri Nov 18 16:05:34 2022

@author: jnguy

min f(x) = -(x_1 - 2)^2

s.t:
    0 = x_0^2 + x_1 -1
    -1 <= x_0 <= 1
   
One constraint
lower bounds and upper bounds 

Expected solution:
[1.0, 0.0]
    
"""



#% Import stuff here
import ipopt
import numpy as np

#% Classes 
class OptExample(object):
    def __init__(self):
        pass
    
    def objective(self, x):
        """objective function"""
        return -(x[1] - 2)**2

    def gradient(self, x):
        """gradient of objective function"""
        return np.array([
            0, 
            -2*(x[1] - 2)])
        
    def constraints(self, x):
        """return cosntraint vector size of num_constraints x 1"""
        constraints = np.array((
            x[0]**2 + x[1] - 1
        ))
        
        print("constraints shape is", constraints)
        
        return constraints
        
    def jacobian(self, x):
        """return jacobian matrix size of num_constraints x num_variables"""
        jacob = np.array([2*x[0], 1])
        print("shape of jacboian is", jacob.shape)
        
        return jacob
    
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
        print("\n")

#% Main 
if __name__ == '__main__':
    
    #initial guesses x0 -
    x0 = [3.5, 2.5]
    
    lb = [-1 , -1] #lower bounds
    ub = [1, 1] #upper bounds
    
    cl = [0]
    cu = [0]
    
    nlp = ipopt.problem(
                n=len(x0),
                m=len(cl),
                # objective=hs071.objective,
                # gradient=hs071.gradient,
                # constraints=hs071.constraints, 
                problem_obj=OptExample(),
                lb=lb,
                ub=ub,
                cl=cl,
                cu=cu
                )

    x, info = nlp.solve(x0)
    
    