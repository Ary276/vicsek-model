import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress

plt.figure(figsize=(10, 10))
data = np.load('./data/V_a_fov.npy')
data = (data[0:50,:] + data[50:100,:] + data[100:150,:] + data[150:200,:] + data[200:250,:] + data[250:300,:] + data[300:350,:] + data[350:400,:] + data[400:450,:] + data[450:500,:])/10
data = np.mean(data[:, -50:], axis=1)
fov = np.linspace(0, 2*np.pi, 50)
plt.plot(fov, data, label='N = 400')

data = np.load('./data/V_a_fov_40.npy')
data = (data[0:50,:] + data[50:100,:] + data[100:150,:] + data[150:200,:] + data[200:250,:] + data[250:300,:] + data[300:350,:] + data[350:400,:] + data[400:450,:] + data[450:500,:])/10
data = np.mean(data[:, -50:], axis=1)
fov = np.linspace(0, 2*np.pi, 50)
plt.plot(fov, data, label='N = 40')

data = np.load('./data/V_a_fov_100.npy')
data = (data[0:50,:] + data[50:100,:] + data[100:150,:] + data[150:200,:] + data[200:250,:] + data[250:300,:] + data[300:350,:] + data[350:400,:] + data[400:450,:] + data[450:500,:])/10
data = np.mean(data[:, -50:], axis=1)
fov = np.linspace(0, 2*np.pi, 50)
plt.plot(fov, data, label='N = 100')

data = np.load('./data/V_a_fov_1000.npy')
data = (data[0:50,:] + data[50:100,:] + data[100:150,:] + data[150:200,:] + data[200:250,:] + data[250:300,:] + data[300:350,:] + data[350:400,:] + data[400:450,:] + data[450:500,:])/10
data = np.mean(data[:, -50:], axis=1)
fov = np.linspace(0, 2*np.pi, 50)
plt.plot(fov, data, label='N = 1000')

plt.xlabel('Field of View' , fontsize = 20)
plt.ylabel('Average Velocity', fontsize = 20)
plt.title('Average Velocity vs Field of View', fontsize = 20)
plt.legend()
plt.savefig('V_a_fov.png')


plt.figure(figsize=(10, 10))
data = np.load('./data/V_a_fov_1000.npy')
data = (data[0:50,:] + data[50:100,:] + data[100:150,:] + data[150:200,:] + data[200:250,:] + data[250:300,:] + data[300:350,:] + data[350:400,:] + data[400:450,:] + data[450:500,:])/10
plt.plot(data[20, :])
print(data[:,-1])


fig = plt.figure(figsize=(10, 10))


ax1 = plt.subplot(2, 2, 1)

N = 300
n = 2.0
L = 7
data = np.load(f'./data/viscek_fov_n={n}N={N}L={L}.npy')
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
data = np.load(f'./data/viscek_fov_n={n}N={N}L={L}.npy')
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
data = np.load(f'./data/viscek_fov_n={n}N={N}L={L}.npy')
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
data = np.load(f'./data/viscek_fov_n={n}N={N}L={L}.npy')
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

plt.savefig('viscek_fov.png')


plt.figure(figsize=(10, 10))
ax = plt.gca()
ax.set_xscale('log')
ax.set_yscale('log')
fov = np.linspace(0, 2*np.pi, 50)
data = np.load('./data/V_a_fov.npy')
print(data.shape)
data = (data[0:50,:] + data[50:100,:] + data[100:150,:] + data[150:200,:] + data[200:250,:] + data[250:300,:] + data[300:350,:] + data[350:400,:] + data[400:450,:] + data[450:500,:])/10
data = np.mean(data[:, -100:], axis=1)
fov_c = 3.1
ind = np.where((-fov_c + fov)/fov_c > 0.1)
ax.scatter(np.abs((fov_c - fov[ind])/fov_c), data[ind], s = 50, marker = '+', label = 'N = 400')
a1, b1, r1, p1, s1 = linregress(np.log(np.abs((fov_c - fov[ind])/fov_c)), np.log(data[ind]))
print(a1, b1, r1, p1, s1)

data = np.load('./data/V_a_fov_40.npy')
print(data.shape)
data = (data[0:50,:] + data[50:100,:] + data[100:150,:] + data[150:200,:] + data[200:250,:] + data[250:300,:] + data[300:350,:] + data[350:400,:] + data[400:450,:] + data[450:500,:])/10
data = np.mean(data[:, -100:], axis=1)
fov_c = 3.1
ind = np.where((-fov_c + fov)/fov_c > 0.1)
ax.scatter(np.abs((fov_c - fov[ind])/fov_c), data[ind], s = 50, marker = '+', label = 'N = 40')
a2, b2, r2, p2, s2 = linregress(np.log(np.abs((fov_c - fov[ind])/fov_c)), np.log(data[ind]))
print(a2, b2, r2, p2, s2)

data = np.load('./data/V_a_fov_100.npy')
print(data.shape)
data = (data[0:50,:] + data[50:100,:] + data[100:150,:] + data[150:200,:] + data[200:250,:] + data[250:300,:] + data[300:350,:] + data[350:400,:] + data[400:450,:] + data[450:500,:])/10
data = np.mean(data[:, -100:], axis=1)
fov_c = 3.1
ind = np.where((-fov_c + fov)/fov_c > 0.1)
ax.scatter(np.abs((fov_c - fov[ind])/fov_c), data[ind], s = 50, marker = '+', label = 'N = 100')
a3, b3, r3, p3, s3 = linregress(np.log(np.abs((fov_c - fov[ind])/fov_c)), np.log(data[ind]))
print(a3, b3, r3, p3, s3)

a = (a1 + a2 + a3 )/3
b = (b1 + b2 + b3 )/3
stderr = (s1 + s2 + s3 )/3
x = np.linspace(0.1, 1, 100)
y1 = np.exp(b1)*x**a1
y2 = np.exp(b2)*x**a2
y3 = np.exp(b3)*x**a3

ax.plot(x, y1, '-', label = 'Power Law Fit', color = 'black')
ax.plot(x, y2, '-', label = 'Power Law Fit', color = 'black')
ax.plot(x, y3, '-', label = 'Power Law Fit', color = 'black')
ax.text(0.4, 0.3, 'slope = ' + str(round(a1, 3)) + '$\pm$' + str(round(s1, 4)) , horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize = 15)
ax.text(0.3, 0.95, 'slope = ' + str(round(a2, 3)) + '$\pm$' + str(round(s2, 4)) , horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize = 15)
ax.text(0.3, 0.6, 'slope = ' + str(round(a3, 3)) + '$\pm$' + str(round(s3, 4)) , horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize = 15)

ax.set_xlabel('$(FOV - FOV_c)/FOV_c$', fontsize = 20)
ax.set_ylabel('$v_a$', fontsize = 20)
ax.set_title('Scaling With FOV', fontsize = 20)
ax.legend(fontsize = 15)
plt.savefig('fov_scaling.png')
plt.show()