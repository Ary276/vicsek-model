import numpy as np
import scipy
import time

start = time.time()
N = 100
x = np.random.uniform(0, 1, (N, 1))
y = np.random.uniform(0, 1, (N, 1))
dat = np.concatenate((x, y), axis=1)
print(dat.shape)
tree = scipy.spatial.cKDTree(dat, boxsize=(1, 1))
i = 15
nearest_neighbour = tree.query_ball_point(dat[i], 0.1)
print(nearest_neighbour)
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
