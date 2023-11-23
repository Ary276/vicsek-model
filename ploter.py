"""
Author : Aryaman Bhutani
S.R. No: 18721
PH 325 Advanced Statistical Physics
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress



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
ax1.quiver(X[:4], Y[:4], np.cos(Theta[:4]), np.sin(Theta[:4]), headwidth=0, headlength=0, headaxislength=0)
ax1.quiver(X[5], Y[5], np.cos(Theta[5]), np.sin(Theta[5]))


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
ax2.set_title(f'(b) N = {N}, L = {L}, $\eta$ = {n}, t = 500')
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
ax3.set_title(f'(b) N = {N}, L = {L}, $\eta$ = {n}, t = 500')
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
ax4.set_title(f'(b) N = {N}, L = {L}, $\eta$ = {n}, t = 500')
ax4.set_aspect('equal')
ax4.quiver(X[-20:-1], Y[-20:-1], np.cos(Theta[-20:-1]), np.sin(Theta[-20:-1]), headwidth=0, headlength=0, headaxislength=0)
ax4.quiver(X[-1], Y[-1], np.cos(Theta[-1]), np.sin(Theta[-1]))

plt.savefig('viscek.png')

plt.figure(figsize=(10, 10))
eta = np.linspace(0, 6, 30)

data = np.load('V_a_0.npy')
print(data.shape)
data = (data[0:30, :] + data[30:60, :] + data[60:90,:] + data[90:120,:] + data[120:150,:])/5
print(data.shape)
data = np.mean(data[:, -100:], axis=1)
plt.plot(eta, data, '-+', ms = 10, label='N = 40, L = 3.1')

data = np.load('V_a_1.npy')
print(data.shape)
data = (data[0:30, :] + data[30:60, :] + data[60:90,:] + data[90:120,:] + data[120:150,:])/5
data = np.mean(data[:, -100:], axis=1)
plt.plot(eta, data, '-v', ms = 10, label='N = 100, L = 3.1')

data = np.load('V_a_2.npy')
print(data.shape)
data = (data[0:30, :] + data[30:60, :] + data[60:90,:] + data[90:120,:] + data[120:150,:])/5
data = np.mean(data[:, -100:], axis=1)
plt.plot(eta, data, '-p', ms = 10, label='N = 400, L = 3.1')

data = np.load('V_a_3.npy')
print(data.shape)
data = (data[0:30, :] + data[30:60, :] + data[60:90,:] + data[90:120,:] + data[120:150,:])/5
data = np.mean(data[:, -100:], axis=1)
plt.plot(eta, data, '-*', ms = 10, label='N = 4000, L = 3.1')

#plt.scatter(eta, data[40:50], label='N = 1000, L = 3.1')
plt.legend(fontsize = 15)
plt.xlabel('$\eta$', fontsize = 20)
plt.ylabel('$v_a$', fontsize = 20)
plt.savefig('V_a.png')

#fig = plt.figure(figsize=(10, 10))

#data = np.load('V_a_rho.npy')
#rho = np.linspace(10, 4000, 50)/(20*20)
#data = np.mean(data[:, -100:], axis=1)
#plt.plot(rho, data, '-+')


plt.figure(figsize=(10, 10))
ax = plt.gca()
ax.set_xscale('log')
ax.set_yscale('log')
eta = np.linspace(0, 5.8, 30)
data = np.load('V_a_0.npy')
print(data.shape)
data = (data[0:30, :] + data[30:60, :] + data[60:90,:] + data[90:120,:] + data[120:150,:])/5
data = np.mean(data[:, -100:], axis=1)
n_c = 3.6
ind = np.where((n_c - eta)/n_c > 0.1)
ax.scatter(np.abs((n_c - eta[ind])/n_c), data[ind], s = 50, marker = '+', label = 'N = 40')
a1, b1, r1, p1, s1 = linregress(np.log(np.abs((n_c - eta[ind])/n_c)), np.log(data[ind]))
print(a1, b1, r1, p1, s1)

data = np.load('V_a_1.npy')
data = (data[0:30, :] + data[30:60, :] + data[60:90,:] + data[90:120,:] + data[120:150,:])/5
data = np.mean(data[:, -100:], axis=1)
n_c = 3.4
ind = np.where((n_c - eta)/n_c > 0.1)
ax.scatter(np.abs((n_c - eta[ind])/n_c), data[ind], s = 50,  marker = 'v',  label = 'N = 100')
a2, b2, r2, p2, s2 = linregress(np.log(np.abs((n_c - eta[ind])/n_c)), np.log(data[ind]))
print(a2, b2, r2, p2, s2)

data = np.load('V_a_2.npy')
data = (data[0:30, :] + data[30:60, :] + data[60:90,:] + data[90:120,:] + data[120:150,:])/5
data = np.mean(data[:, -100:], axis=1)
n_c = 3.2
ind = np.where((n_c - eta)/n_c > 0.1)
ax.scatter(np.abs((n_c - eta[ind])/n_c), data[ind], s = 50, marker = 'p', label = 'N = 400')
a3, b3, r3, p3, s3 = linregress(np.log(np.abs((n_c - eta[ind])/n_c)), np.log(data[ind]))
print(a3, b3, r3, p3, s3)

data = np.load('V_a_3.npy')
data = (data[0:30, :] + data[30:60, :] + data[60:90,:] + data[90:120,:] + data[120:150,:])/5
data = np.mean(data[:, -100:], axis=1)
n_c = 3.0
ind = np.where((n_c - eta)/n_c > 0.1)
ax.scatter(np.abs((n_c - eta[ind])/n_c), data[ind], s = 50,  marker = '*', label = 'N = 4000')
a4, b4, r4, p4, s4 = linregress(np.log(np.abs((n_c - eta[ind])/n_c)), np.log(data[ind]))
print(a4, b4, r4, p4, s4)

a = (a1 + a2 + a3 + a4)/4
b = (b1 + b2 + b3 + b4)/4
stderr = (s1 + s2 + s3 + s4)/4
x = np.linspace(0.1, 1, 100)
y = np.exp(b)*x**a

ax.plot(x, y, '-', label = 'Power Law Fit', color = 'black')

ax.text(0.3, 0.7, 'slope = ' + str(round(a, 2)) + '$\pm$' + str(round(stderr, 3)) + r' for $\rho$ = 4', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize = 15)

ax.set_xlabel('$(\eta_c(L) - \eta)/\eta_c(L)$', fontsize = 20)
ax.set_ylabel('$v_a$', fontsize = 20)

ax.legend(fontsize = 15)

plt.savefig('eta.png')

plt.figure(figsize=(10, 10))

data = np.load('V_a_rho.npy')
data = (data[0:50, :] + data[50:100, :] + data[100:150,:] + data[150:200,:] + data[200:250,:] + data[250:300, :] + data[300:350, :] + data[350:400, :] + data[400:450,:] + data[450:500,:])/10
data = np.mean(data[:, -100:], axis=1)
rho = np.logspace(1.6, 3.6, 50)/(20*20)
plt.scatter(rho, data, s = 100 , marker = '+' , label = 'L = 20, $\eta$ = 2')
plt.xlabel(r'$\rho$', fontsize = 20)
plt.ylabel('$v_a$', fontsize = 20)
plt.legend(fontsize = 15)
plt.savefig('v_a_rho.png')

plt.figure( figsize=(10, 10))
ax = plt.gca()
rho_c = 0.16
ind = np.where((rho - rho_c)/rho_c > 0.4)
ax.scatter(np.abs((rho[ind] - rho_c)/rho_c), data[ind], s = 100, marker = '+', label = 'L = 20, $\eta$ = 2')
a, b, r, p, s = linregress(np.log(np.abs((rho[ind] - rho_c)/rho_c)), np.log(data[ind]))
print(a, b, r, p, s)
x = np.linspace((rho[ind][0] - rho_c)/rho_c, (rho[ind][-1]-rho_c)/rho_c, 100)
y = np.exp(b)*x**a
ax.plot(x, y, '-', label = 'Power Law Fit', color = 'black')
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel(r'$(\rho - \rho_c)/\rho_c$', fontsize = 20)
ax.set_ylabel('$v_a$', fontsize = 20)
ax.text(0.3, 0.8, 'slope = ' + str(round(a, 2)) + '$\pm$' + str(round(s, 3)) + r' for $\eta$ = 2, L = 20', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize = 15)

plt.savefig('rho.png')
