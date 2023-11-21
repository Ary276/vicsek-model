"""
Author : Aryaman Bhutani
S.R. No: 18721
PH 325 Advanced Statistical Physics
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# animating the viscek model

# reading the data from the file
data = np.load('./data/viscek_n=0.1N=300.npy')
print(data.shape)
X = data[0]
Y = data[1]
Theta = data[2]

# reading the parameters from the file
params = np.loadtxt('params.txt', delimiter=',')
params = params[0]
N = params[0]
v = params[1]
L = params[2]
dt = params[3]
t_max = params[4]
R = params[5]
eta = params[6]
seed = params[7]

# animating the viscek model using matplotlib.animation quiver plot
fig = plt.figure()
ax = plt.axes(xlim=(0, L), ylim=(0, L))
ax.set_title('Viscek Model')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_aspect('equal')
ax.quiver(X[0], Y[0], np.cos(Theta[0]), np.sin(Theta[0]))
def animate(i):
    ax.clear()
    ax.set_xlim(0, L)
    ax.set_ylim(0, L)
    ax.set_title('Viscek Model  t = ' + str(i*dt) + 's')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_aspect('equal')
    ax.quiver(X[i], Y[i], np.cos(Theta[i]), np.sin(Theta[i]))
    return ax

anim = animation.FuncAnimation(fig, animate, frames=int(t_max/dt), interval=1)
anim.save('viscek.mp4', writer='ffmpeg', fps=10)

fig = plt.figure()
ax = plt.axes(xlim=(0, L), ylim=(0, L))
ax.set_title('Viscek Model')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_aspect('equal')

for i in range(data.shape[2]):
    ax.plot(X[-20:, i], Y[-20:, i], '.', markersize=1, color='black')
ax.quiver(X[-1], Y[-1], np.cos(Theta[-1]), np.sin(Theta[-1]))

plt.savefig('viscek.png')