import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

plt.figure(figsize=(10, 10))

# reading the data from the file
data = np.load('V_a_.npy')
print(data.shape)

plt.figure(1)
for i in range(50):
    plt.plot(data[i,:])
plt.show()

plt.figure(2)
plt.plot(np.mean(data[0:50, :-10], axis=1))
plt.plot(np.mean(data[50:100, :-10], axis=1))
plt.plot(np.mean(data[100:150, :-10], axis=1))
plt.plot(np.mean(data[150:200, :-10], axis=1))
plt.plot(np.mean(data[200:250, :-10], axis=1))
plt.plot(np.mean(data[250:300, :-10], axis=1))
plt.plot(np.mean(data[300:350, :-10], axis=1))
plt.plot(np.mean(data[350:400, :-10], axis=1))
plt.plot(np.mean(data[400:450, :-10], axis=1))
plt.plot(np.mean(data[450:500, :-10], axis=1))
plt.show()
