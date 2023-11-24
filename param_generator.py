import numpy as np
import time

# initial values of the parameters
N = 40
v = 0.03
L = 3.1
dt = 1
t_max = 500
R = 1
eta = 2.4
seed = 0
fov = np.linspace(0, 2*np.pi, 50)

n = 500
params = np.zeros((n, 9))
for i in range(n):
    params[i, 0] = N
    params[i, 1] = v
    params[i, 2] = L
    params[i, 3] = dt
    params[i, 4] = t_max
    params[i, 5] = R
    params[i, 6] = 1
    params[i, 7] = 0
    params[i, 8] = fov[i%50]
    
np.savetxt('params.txt', params, delimiter=',')


