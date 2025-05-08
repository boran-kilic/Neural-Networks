import numpy as np
import matplotlib.pyplot as plt
import gzip

def load_mnist_images(filename):
    with gzip.open(filename, 'rb') as f:
        f.read(16)
        buffer = f.read()
        images = np.frombuffer(buffer, dtype=np.uint8)
        images = images.reshape(-1, 784) 
        images = images.astype('float32') / 255.0 
    return images

def load_mnist_labels(filename):
    with gzip.open(filename, 'rb') as f:
        f.read(8)
        buffer = f.read()
        labels = np.frombuffer(buffer, dtype=np.uint8)
    return labels

def step_function(x):
    return np.where(x > 0, 1, 0)

def train_multicategory_pta(W, train_images, train_labels, n, eta, epsilon):
    Wcopy = W.copy()
    epoch = 0
    errors = []

    while True:
        epoch_errors = 0
        for i in range(n):
            v = np.dot(Wcopy, train_images[i])
            predicted_label = np.argmax(v)
            actual_label = train_labels[i]

            if predicted_label != actual_label:
                epoch_errors += 1

        errors.append(epoch_errors)
        # print(f"Epoch {epoch}: {epoch_errors} misclassifications")

        if (epoch > 0 and errors[-1] / n <= epsilon) or epoch >= 50:
            break

        for i in range(n):
            v = np.dot(Wcopy, train_images[i])
            output = step_function(v)
            desired_output = np.zeros(10)
            desired_output[train_labels[i]] = 1
            Wcopy += eta * (desired_output - output).reshape(-1, 1) * train_images[i]

        epoch += 1
    
    plt.figure()    
    plt.plot(errors)
    plt.xlabel('Epoch')
    plt.ylabel('Misclassification Errors')
    plt.title(f'Epoch vs Misclassification Errors n = {n}, η = {eta}, ε = {epsilon}')
    
    return Wcopy, errors

def test_multicategory_pta(Wtrain, test_images, test_labels):
    n = test_images.shape[0]
    errors = 0

    for i in range(n):
        v = np.dot(Wtrain, test_images[i])
        predicted_label = np.argmax(v)
        actual_label = test_labels[i]

        if predicted_label != actual_label:
            errors += 1

    return errors

train_images = load_mnist_images('train-images-idx3-ubyte.gz')
train_labels = load_mnist_labels('train-labels-idx1-ubyte.gz')
test_images = load_mnist_images('t10k-images-idx3-ubyte.gz')
test_labels = load_mnist_labels('t10k-labels-idx1-ubyte.gz')

print("Training images shape:", train_images.shape)
print("Training labels shape:", train_labels.shape)
print("Test images shape:", test_images.shape)
print("Test labels shape:", test_labels.shape)

np.random.seed(29)

plt.ion()

W = np.random.randn(10, 784) * 0.01

Wtrain, errors = train_multicategory_pta(W, train_images, train_labels, n = 50, eta = 1, epsilon = 0)
error = test_multicategory_pta(Wtrain, test_images, test_labels)
percentage_error = error/100
print(f'n = 50, η = 1, ε = 0, Test Error: {percentage_error}%')

Wtrain, errors = train_multicategory_pta(W, train_images, train_labels, n = 1000, eta = 1, epsilon = 0)
error = test_multicategory_pta(Wtrain, test_images, test_labels)
percentage_error = error/100
print(f'n = 1000, η = 1, ε = 0, Test Error: {percentage_error}%')

Wtrain, errors = train_multicategory_pta(W,train_images, train_labels, n = 60000, eta = 1, epsilon = 0)
error = test_multicategory_pta(Wtrain, test_images, test_labels)
percentage_error = error/100
print(f'n = 60000, η = 1, ε = 0, Test Error: {percentage_error}%')

eta = 0.5   
epsilon = 0.142
Wtrain, errors = train_multicategory_pta(W,train_images, train_labels, n=60000, eta=eta, epsilon=epsilon)
error = test_multicategory_pta(Wtrain, test_images, test_labels)
percentage_error = error/100
print(f'n = 60000, η = {eta}, ε = {epsilon}, Test Error: {percentage_error}%')

for i in range(3):
    print("Iteration:", i+1)
    W = np.random.randn(10, 784) * 0.01
    Wtrain, errors = train_multicategory_pta(W,train_images, train_labels, n = 60000, eta = 0.5, epsilon = 0.142)
    error = test_multicategory_pta(Wtrain, test_images, test_labels)
    percentage_error = error/100
    print(f'n = 60000, η = 0.5, ε = 0.142, Test Error: {percentage_error}%')


plt.ioff()
plt.show()
