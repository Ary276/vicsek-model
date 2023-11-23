"""
Author : Aryaman Bhutani
S.R. No: 18721
PH 325 Advanced Statistical Physics
"""
import numpy as np
import time
import scipy
import multiprocessing

# Defining the function to calculate the distance between two particles with periodic boundary conditions

def distance(x1, y1, x2, y2, *params):
    L = params[2]
    if np.abs(y1-y2) < L/2:
        dy = np.abs(y1-y2)
    else:
        dy = L - np.abs(y1-y2)
    
    if np.abs(x1-x2) < L/2:
        dx = np.abs(x1-x2)
    else:
        dx = L - np.abs(x1-x2)
    return np.sqrt(dx**2 + dy**2)

# Calculating mean theta value


def mean_theta(x_in, y_in, theta_in, *params):
    N = int(params[0])
    R = params[5]
    fov = params[8]
    mean_theta = np.zeros((N, 1))
    coord = np.concatenate((x_in, y_in), axis=1)
    tree = scipy.spatial.cKDTree(coord, boxsize=(params[2], params[2]))
    for i in range(N):
        nearest_neighbour = tree.query_ball_point(coord[i], R)
        del_theta = np.abs(theta_in[nearest_neighbour] - theta_in[i])
        nearest_neighbour = (del_theta > fov/2).nonzero()[0]
        avg_sin = np.mean(np.sin(theta_in[nearest_neighbour]))
        avg_cos = np.mean(np.cos(theta_in[nearest_neighbour]))
        mean_theta[i] = np.arctan2(avg_sin, avg_cos)
    return mean_theta

# Updating the positions and velocities of the particles

def update(x, y, theta, *params):
    N = int(params[0])
    v = params[1]
    L = params[2]
    dt = params[3]
    eta = params[6]
    vx = v*np.cos(theta)
    vy = v*np.sin(theta)
    theta = (mean_theta(x, y, theta, *params) + np.random.uniform(-eta/2, eta/2, (N, 1)))
    x = (x + vx*dt)%L
    y = (y + vy*dt)%L
    return x, y, theta


def compute(*params):
    # Reading the parameters from the file
    N = int(params[0])
    v = params[1]
    L = params[2]
    dt = params[3]
    t_max = params[4]
    R = params[5]
    eta = params[6]
    seed = int(params[7])
    #np.random.seed(seed)

    # Defining the initial positions and velocities
    x = np.random.uniform(0, L, (N, 1))
    y = np.random.uniform(0, L, (N, 1))
    theta = np.random.uniform(-np.pi, np.pi, (N, 1))
    X = np.zeros((int(t_max/dt), N))
    Y = np.zeros((int(t_max/dt), N))
    Theta = np.zeros((int(t_max/dt), N))

    # Calculating the positions and velocities of the particles at different times
    t = 0
    for i in range(int(t_max/dt)):
        X[i, :] = x.reshape(N)
        Y[i, :] = y.reshape(N)
        Theta[i, :] = theta.reshape(N)
        x, y, theta = update(x, y, theta, *params)
        t = t + dt

    # Saving the data in a file
    data = np.stack((X, Y, Theta))
    return data


# Reading the parameters from the file
param_list = np.loadtxt('params.txt', delimiter=',')


def vel(i):
    param = param_list[i]
    data = compute(*param)
    #np.save(f'./data/viscek_n={param[6]:0,.1f}N={int(param[0])}L={int(param[2])}.npy', data)
    Vx = np.cos(data[2, :, :])
    Vy = np.sin(data[2, :, :])
    V = np.stack((Vx, Vy), axis=0)
    V = np.sum(V, axis=2)
    v = np.sqrt(np.sum(V**2, axis=0))/param[0]
    #print('Average Velocity = ', v)
    return v

if __name__ == '__main__':
    start = time.time()
    print(param_list.shape)
    iters = int(param_list.shape[0])
    n_proc = multiprocessing.cpu_count()
    print('Number of processors = ', n_proc)
    with multiprocessing.Pool(processes=n_proc) as pool:
        V_a = pool.map(vel, list(range(iters)))
end = time.time()
#np.save(f'V_a_rho.npy', V_a)
print('Time taken = ', end-start, 's')