import numpy as np

# initial values of the parameters
N = [40, 100, 400, 4000, 10000]
v = 0.03
L = [3.1, 5, 10, 31.6, 50]
dt = 1
t_max = 200
R = 1
eta = 3

n = 50
params = np.zeros((n, 7))
for i in range(n):
    params[i, 0] = N[3]
    params[i, 1] = v
    params[i, 2] = L[3]
    params[i, 3] = dt
    params[i, 4] = t_max
    params[i, 5] = R
    params[i, 6] = (0.1*i)%5
    
np.savetxt('params.txt', params, delimiter=',')


