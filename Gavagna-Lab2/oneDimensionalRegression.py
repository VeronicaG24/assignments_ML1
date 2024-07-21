# ----------------------------------------------------------------------------------------------------------------------
# One-Dimensional Problem
# ----------------------------------------------------------------------------------------------------------------------
# Veronica Gavagna
# ----------------------------------------------------------------------------------------------------------------------
#

##############################################################
# Compute one-dimensional problem without intercept
# RETURN:
#   least squares solution
##############################################################
def compute_least_squares_sol_noIntercept(ds_turkish, obs_index, target_index):
    num = 0.
    den = 0.
    for i in range(0, ds_turkish.shape[0], 1):
        num = num + ds_turkish[i, obs_index] * ds_turkish[i, target_index]
        den = den + ds_turkish[i, obs_index] * ds_turkish[i, obs_index]

    w = num / den
    return w


##############################################################
# Compute one-dimensional problem with intercept
# RETURN:
#   intercept and slop
##############################################################
def compute_least_squares_sol_intercept(ds_mtcars, obs_index, target_index):
    x_mean = 0.
    t_mean = 0.
    w1_num = 0.
    w1_den = 0.
    for i in range(0, ds_mtcars.shape[0], 1):
        x_mean = x_mean + ds_mtcars[i, obs_index]
        t_mean = t_mean + ds_mtcars[i, target_index]

    x_mean = x_mean / ds_mtcars.shape[0]
    t_mean = t_mean / ds_mtcars.shape[0]

    for j in range(0, ds_mtcars.shape[0], 1):
        w1_num = w1_num + ((ds_mtcars[j, obs_index] - x_mean) * (ds_mtcars[j, target_index] - t_mean))
        w1_den = w1_den + ((ds_mtcars[j, obs_index] - x_mean) * (ds_mtcars[j, obs_index] - x_mean))

    w1 = w1_num / w1_den
    w0 = t_mean - (w1 * x_mean)

    return w0, w1


##############################################################
# Compute mean squares error without intercept
# RETURN:
#   mean squares error
##############################################################
def mean_square_error_function_noIntercept(ds, obsIndex, targetIndex, w):
    den_j = ds.shape[0]
    num_j = 0.
    for i in range(0, ds.shape[0], 1):
        y = w * ds[i, obsIndex]
        num_j = num_j + ((y - ds[i, targetIndex]) * (y - ds[i, targetIndex]))

    j = num_j / den_j

    return j


##############################################################
# Compute mean squares error with intercept
# RETURN:
#   mean squares error
##############################################################
def mean_square_error_function_intercept(ds, obsIndex, targetIndex, w0, w1):
    den_j = ds.shape[0]
    num_j = 0.
    for i in range(0, ds.shape[0], 1):
        y = w0 + (w1 * ds[i, obsIndex])
        num_j = num_j + ((y - ds[i, targetIndex]) * (y - ds[i, targetIndex]))

    j = num_j / den_j

    return j
