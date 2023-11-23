import numpy as np

# initial values of the parameters
N = np.logspace(1, 3.6, 30)
v = 0.03
L = 20
dt = 1
t_max = 500
R = 1
eta = 2.4
seed = 0

n = 150
params = np.zeros((n, 8))
for i in range(n):
    params[i, 0] = int(N[i%30])
    params[i, 1] = v
    params[i, 2] = L
    params[i, 3] = dt
    params[i, 4] = t_max
    params[i, 5] = R
    params[i, 6] = 3
    params[i, 7] = i//30
    
np.savetxt('params.txt', params, delimiter=',')


