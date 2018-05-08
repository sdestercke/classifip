import numpy as np
from cvxopt import solvers, matrix, normal


def data_test(d):
		e_mean = np.random.normal(size=d)
		e_cov = normal(d, d)
		e_cov = e_cov.T * e_cov
		query = np.random.normal(size=d)
		return e_mean, e_cov, query


A = np.matrix([[4.00e-01, -6.08e-01, 9.21e-01, 8.81e-01, -1.96e-01],
															[-6.08e-01, 1.06e+01, -1.59e+01, -1.33e+01, 8.07e+00],
															[9.21e-01, -1.59e+01, 2.44e+01, 2.00e+01, -1.22e+01],
															[8.81e-01, -1.33e+01, 2.00e+01, 1.75e+01, -1.05e+01],
															[-1.96e-01, 8.07e+00, -1.22e+01, -1.05e+01, 7.17e+00]])

b = np.array([-7.02e-01, 7.36e+00, -1.10e+01, -9.97e+00, 6.12e+00])
lower = np.array([-6.33e-01, -2.76e+00, -1.36e+00, 3.57e-01, -1.50e+00])
upper = np.array([1.37e+00, -7.64e-01, 6.40e-01, 2.36e+00, 4.97e-01])

import cvxpy as cvx
from qcqp import *

n = 5
# Form a nonconvex problem.
x = cvx.Variable(n)
cons = [x <= upper, x >= lower]
prob = cvx.Problem(cvx.Minimize(x.T*A*x + b.T*x), cons)

# Create a QCQP handler.
qcqp = QCQP(prob)

# Solve the SDP relaxation and get a starting point to a local method
qcqp.suggest(SDR)
print("SDR lower bound: %.3f" % qcqp.sdr_bound)

# Attempt to improve the starting point given by the suggest method
f_cd, v_cd = qcqp.improve(COORD_DESCENT)
print("Coordinate descent: objective %.3f, violation %.3f" % (f_cd, v_cd))
print(x.value)