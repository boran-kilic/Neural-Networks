import numpy as np
import matplotlib.pyplot as plt

plt.ion()
np.random.seed(67)

w0 = np.random.uniform(-1/4, 1/4)
w1 = np.random.uniform(-1, 1)
w2 = np.random.uniform(-1, 1)
print(f"Optimal Weights w0: {w0}, w1: {w1}, w2: {w2}")


def perceptron_training(X, w, labels, eta, max_epochs=1000):
    wnew= w
    misclassifications = []
    for epoch in range(max_epochs):
        errors = 0
        for i in range(len(X)):
            x = np.array([1, X[i][0], X[i][1]]) 
            y = 1 if labels[i] == 1 else 0
            y_pred = 1 if np.dot(wnew, x) >= 0 else 0
            
            if y_pred != y:
                errors += 1
                wnew += eta * (y - y_pred) * x  
        
        misclassifications.append(errors)
        if errors == 0:
            break  

    return wnew, misclassifications

etas = [1, 10, 0.1]
ns = [100, 1000]
initial_weights = np.random.uniform(-1, 1, 3)
print(f"Initial Weights w0'={initial_weights[0]}, w1'={initial_weights[1]}, w2'={initial_weights[2]}")

for n in ns:
    print(f'n = {n}')
    X = np.random.uniform(-1, 1, (n, 2))

    S1 = []  
    S0 = []  
    for x1, x2 in X:
        if w0 + w1 * x1 + w2 * x2 >= 0:
            S1.append([x1, x2])
        else:
            S0.append([x1, x2])
    S1 = np.array(S1)
    S0 = np.array(S0)

    plt.figure(figsize=(8, 6))
    x_vals = np.linspace(-1, 1, n)
    y_vals = (-w0 - w1 * x_vals) / w2  
    plt.plot(x_vals, y_vals, 'k-', label="Decision Boundary")
    plt.ylim(-1, 1)
    if len(S1) > 0:
        plt.scatter(S1[:, 0], S1[:, 1], marker='o', label='Class S1')
    if len(S0) > 0:
        plt.scatter(S0[:, 0], S0[:, 1], marker='x', label='Class S0')
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.legend()
    plt.title(f"Initial Data Distribution for n={n}")
    plt.pause(0.001)

    labels = [1 if w0 + w1 * x1 + w2 * x2 >= 0 else 0 for x1, x2 in X]


    for eta in etas:
        final_weights = None
        misclassifications = None
        weights_copy = initial_weights.copy() 
        plt.figure(figsize=(8, 6))
        final_weights, misclassifications = perceptron_training(X, weights_copy, labels, eta)
        plt.plot(range(len(misclassifications)), misclassifications, label=f"eta={eta}")
        plt.xlabel("Epoch")
        plt.ylabel("Misclassifications")
        plt.legend()
        plt.title(f"Epoch vs Misclassification Count for eta={eta} and n={n}")
        plt.pause(0.001)
        print(f"Final Weights eta={eta} w0={final_weights[0]}, w1={final_weights[1]}, w2={final_weights[2]}")

plt.ioff()
plt.show()