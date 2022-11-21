# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 23:17:56 2022

@author: jnguy

https://coin-or.github.io/Ipopt/INTERFACES.html
"""

#% Import stuff here
import ipopt
import numpy as np
from scipy.sparse import coo_matrix


#% Classes 

class hs071(object):
    def __init__(self):
        pass

    def objective(self, x):
        #
        # The callback for calculating the objectiveiu
        #
        return x[0] * x[3] * np.sum(x[0:3]) + x[2]

    def gradient(self, x):
        #
        # The callback for calculating the gradient
        #
        self.array = np.array([
                    x[0] * x[3] + x[3] * np.sum(x[0:3]),
                    x[0] * x[3],
                    x[0] * x[3] + 1.0,
                    x[0] * np.sum(x[0:3])
                    ])
        
        print("graident array" , self.array.shape)
        return np.array([
                    x[0] * x[3] + x[3] * np.sum(x[0:3]),
                    x[0] * x[3],
                    x[0] * x[3] + 1.0,
                    x[0] * np.sum(x[0:3])
                    ])

    def constraints(self, x):
        #
        # The callback for calculating the constraints
        # num_constraints x 1
        self.constraints = np.array((np.prod(x), np.dot(x, x)))
        print("constraints ", self.constraints.shape)
        return self.constraints

    def jacobian(self, x):
        #
        # The callback for calculating the Jacobian, becomes a 
        # [num_paramers * num_states, 1] row vector
        #
        jacob = np.concatenate((np.prod(x) / x, 2*x))
        print("jacob shape ", jacob.shape)
        return np.concatenate((np.prod(x) / x, 2*x))

    # def hessianstructure(self):
    #     #
    #     # The structure of the Hessian
    #     # Note:
    #     # The default hessian structure is of a lower triangular matrix. Therefore
    #     # this function is redundant. I include it as an example for structure
    #     # callback.
    #     #
    #     global hs

    #     hs = coo_matrix(np.tril(np.ones((4, 4))))
    #     return (hs.col, hs.row)

    # def hessian(self, x, lagrange, obj_factor):
    #     #
    #     # The callback for calculating the Hessian
    #     #
    #     H = obj_factor*np.array((
    #             (2*x[3], 0, 0, 0),
    #             (x[3],   0, 0, 0),
    #             (x[3],   0, 0, 0),
    #             (2*x[0]+x[1]+x[2], x[0], x[0], 0)))

    #     H += lagrange[0]*np.array((
    #             (0, 0, 0, 0),
    #             (x[2]*x[3], 0, 0, 0),
    #             (x[1]*x[3], x[0]*x[3], 0, 0),
    #             (x[1]*x[2], x[0]*x[2], x[0]*x[1], 0)))

    #     H += lagrange[1]*2*np.eye(4)

    #     #
    #     # Note:
    #     #
    #     #
    #     return H[hs.row, hs.col]

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


#% Main 

x0 = [1.0, 5.0, 5.0, 1.0]

lb = [1.0, 1.0, 1.0, 1.0]
ub = [5.0, 5.0, 5.0, 5.0]

#view this as top down 
"""
[low_c_0, low_c_1,
 upp_c_0, upp_c_1
]

constraint c_0 has a lower bound of 25 and upper bound of inf
constraint c_1 is an equality constraint low bound and upp bound same

"""
cl = [25.0, 40.0] # 
cu = [2.0e19, 40.0] #inequality constraints

nlp = ipopt.problem(
            n=len(x0),
            m=len(cl),
            # objective=hs071.objective,
            # gradient=hs071.gradient,
            # constraints=hs071.constraints, 
            problem_obj=hs071(),
            lb=lb,
            ub=ub,
            cl=cl,
            cu=cu
            )

x, info = nlp.solve(x0)
