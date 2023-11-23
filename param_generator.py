import numpy as np

# initial values of the parameters
N = 1000
v = 0.03
L = 50
dt = 1
t_max = 1000
R = 1
eta = 3
seed = 0

n = 150
params = np.zeros((n, 8))
for i in range(n):
    params[i, 0] = int(N)
    params[i, 1] = v
    params[i, 2] = L
    params[i, 3] = dt
    params[i, 4] = t_max
    params[i, 5] = R
    params[i, 6] = (0.2*i)%30
    params[i, 7] = i//30
    
np.savetxt('params.txt', params, delimiter=',')


