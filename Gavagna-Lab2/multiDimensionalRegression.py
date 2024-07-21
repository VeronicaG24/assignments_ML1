# ----------------------------------------------------------------------------------------------------------------------
# Multi-Dimensional Problem
# ----------------------------------------------------------------------------------------------------------------------
# Veronica Gavagna
# ----------------------------------------------------------------------------------------------------------------------
#

import numpy as np
from scipy import linalg


##############################################################
# Compute multi-dimensional problem with intercept
# RETURN:
#   mean squares error matrix
##############################################################
def compute_least_square_matrix(x_matrix, t_matrix):
    x_pseudoinverse = linalg.pinv(x_matrix, rcond=1e-15)
    w_matrix = np.dot(x_pseudoinverse, t_matrix)

    return w_matrix


##############################################################
# Compute mean squares error multi-dimensional problem
# RETURN:
#   mean squares error
##############################################################
def mean_square_error_function(x_matrix, w_matrix, t_matrix, y_pred):
    norm1 = np.dot(x_matrix, w_matrix)
    mem1 = np.linalg.norm(norm1)
    mem1 = (mem1 * mem1) / 2

    mem2 = np.linalg.multi_dot([w_matrix.T, x_matrix.T, t_matrix])
    mem3 = np.linalg.norm(t_matrix)
    mem3 = (mem3 * mem3) / 2
    j = mem1 - mem2 + mem3

    return j
