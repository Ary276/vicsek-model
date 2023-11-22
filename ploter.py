"""
Author : Aryaman Bhutani
S.R. No: 18721
PH 325 Advanced Statistical Physics
"""

import matplotlib.pyplot as plt
import numpy as np




fig = plt.figure(figsize=(10, 10))


ax1 = plt.subplot(2, 2, 1)

N = 300
n = 2.0
L = 7
data = np.load(f'./data/viscek_n={n}N={N}L={L}.npy')
print(data.shape)
X = data[0]
Y = data[1]
Theta = data[2]

ax1.set_xlim(0, L)
ax1.set_ylim(0, L)
ax1.set_title(f'(a) N = {N}, L = {L}, $\eta$ = {n}, t = 0')
ax1.set_aspect('equal')
ax1.quiver(X[:19], Y[:19], np.cos(Theta[:19]), np.sin(Theta[:19]), headwidth=0, headlength=0, headaxislength=0)
ax1.quiver(X[20], Y[20], np.cos(Theta[20]), np.sin(Theta[20]))


ax2 = plt.subplot(2, 2, 2)

N = 300
n = 0.1
L = 25
data = np.load(f'./data/viscek_n={n}N={N}L={L}.npy')
print(data.shape)
X = data[0]
Y = data[1]
Theta = data[2]

ax2.set_xlim(0, L)
ax2.set_ylim(0, L)
ax2.set_title(f'(b) N = {N}, L = {L}, $\eta$ = {n}, t = 200')
ax2.set_aspect('equal')
ax2.quiver(X[-20:-1], Y[-20:-1], np.cos(Theta[-20:-1]), np.sin(Theta[-20:-1]), headwidth=0, headlength=0, headaxislength=0)
ax2.quiver(X[-1], Y[-1], np.cos(Theta[-1]), np.sin(Theta[-1]))


ax3 = plt.subplot(2, 2, 3)

N = 300
n = 2.0
L = 7
data = np.load(f'./data/viscek_n={n}N={N}L={L}.npy')
print(data.shape)
X = data[0]
Y = data[1]
Theta = data[2]

ax3.set_xlim(0, L)
ax3.set_ylim(0, L)
ax3.set_title(f'(b) N = {N}, L = {L}, $\eta$ = {n}, t = 200')
ax3.set_aspect('equal')
ax3.quiver(X[-20:-1], Y[-20:-1], np.cos(Theta[-20:-1]), np.sin(Theta[-20:-1]), headwidth=0, headlength=0, headaxislength=0)
ax3.quiver(X[-1], Y[-1], np.cos(Theta[-1]), np.sin(Theta[-1]))


ax4 = plt.subplot(2, 2, 4)

N = 300
n = 0.1
L = 5
data = np.load(f'./data/viscek_n={n}N={N}L={L}.npy')
print(data.shape)
X = data[0]
Y = data[1]
Theta = data[2]

ax4.set_xlim(0, L)
ax4.set_ylim(0, L)
ax4.set_title(f'(b) N = {N}, L = {L}, $\eta$ = {n}, t = 200')
ax4.set_aspect('equal')
ax4.quiver(X[-20:-1], Y[-20:-1], np.cos(Theta[-20:-1]), np.sin(Theta[-20:-1]), headwidth=0, headlength=0, headaxislength=0)
ax4.quiver(X[-1], Y[-1], np.cos(Theta[-1]), np.sin(Theta[-1]))

plt.savefig('viscek.png')