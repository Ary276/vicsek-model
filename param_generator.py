import numpy as np

# initial values of the parameters
N = 0
v = 0.3
L = [3.1, 5, 10, 31.6, 50]
dt = 1
t_max = 1000
R = 1
eta = 3
seed = 0

n = 12
params = np.zeros((n, 8))
for i in range(n):
    params[i, 0] = 2**i
    params[i, 1] = v
    params[i, 2] = 20
    params[i, 3] = dt
    params[i, 4] = t_max
    params[i, 5] = R
    params[i, 6] = 1
    params[i, 7] = 0
    
np.savetxt('params.txt', params, delimiter=',')


