"""
Author : Aryaman Bhutani
S.R. No: 18721
PH 325 Advanced Statistical Physics
"""

import matplotlib.pyplot as plt
import numpy as np

N = 300
n = 0.1
L = 25
data = np.load(f'./data/viscek_n={n}N={N}L={L}.npy')
print(data.shape)
X = data[0]
Y = data[1]
Theta = data[2]


fig = plt.figure(figsize=(10, 10))
ax = plt.axes(xlim=(0, L), ylim=(0, L))
ax.set_title('Viscek Model')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_aspect('equal')


ax.quiver(X[-20:-1], Y[-20:-1], np.cos(Theta[-20:-1]), np.sin(Theta[-20:-1]), headwidth=0, headlength=0, headaxislength=0)
ax.quiver(X[-1], Y[-1], np.cos(Theta[-1]), np.sin(Theta[-1]))

plt.savefig('viscek.png')