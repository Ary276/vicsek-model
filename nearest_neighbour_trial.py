import numpy as np
import scipy
import time

start = time.time()
N = 100
x = np.random.uniform(0, 1, (N, 1))
y = np.random.uniform(0, 1, (N, 1))
fov = 0.1
theta = np.random.uniform(-np.pi, np.pi, (N, 1))
dat = np.concatenate((x, y), axis=1)
print(dat.shape)
tree = scipy.spatial.cKDTree(dat, boxsize=(1, 1))
i = 15
nearest_neighbour = np.array(tree.query_ball_point(dat[i], 0.1))
print(nearest_neighbour)
angles = np.arctan2(x[nearest_neighbour] - x[i], y[nearest_neighbour] - y[i])
print(angles)
nearest_neighbour_ind = (np.abs(angles) < fov/2).nonzero()[0]

print(nearest_neighbour[nearest_neighbour_ind])
end = time.time()
print('Time taken = ', end-start)
print('-----------------')
start = time.time()

def distance(x1, y1, x2, y2, *params):
    if np.abs(y1-y2) < 1/2:
        dy = np.abs(y1-y2)
    else:
        dy = 1 - np.abs(y1-y2)
    
    if np.abs(x1-x2) < 1/2:
        dx = np.abs(x1-x2)
    else:
        dx = 1 - np.abs(x1-x2)
    return np.sqrt(dx**2 + dy**2)

nn = []
for i in range(N):
    for j in range(N):
        if distance(x[i], y[i], x[j], y[j], ) < 0.1:
            nn.append(j)
print(nn)
end = time.time()
print('Time taken = ', end-start)
