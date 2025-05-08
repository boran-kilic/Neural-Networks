import numpy as np
import matplotlib.pyplot as plt

def step(x):
    if x > 0:
        return 1
    else:
        return 0

def nn(x, y):
    h1 = step(x - y + 1)
    h2 = step(-x - y + 1)
    h3 = step(-x)
    z = step(h1 + h2 - h3 - 1.5)
    return z

x_vals = np.random.uniform(-2, 2, 1000)
y_vals = np.random.uniform(-2, 2, 1000)

outputs = np.array([nn(x, y) for x, y in zip(x_vals, y_vals)])

plt.figure(figsize=(6,6))
plt.scatter(x_vals[outputs == 0], y_vals[outputs == 0], color='blue', label='Output 0')
plt.scatter(x_vals[outputs == 1], y_vals[outputs == 1], color='red', label='Output 1')
plt.xlabel("x")
plt.ylabel("y")
plt.title("Decision Region of the Neural Network")
plt.legend()
plt.grid(True)
plt.show()
