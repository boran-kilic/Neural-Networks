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

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True)) 
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

def cross_entropy_loss(y_true, y_pred):
    return -np.sum(y_true * np.log(y_pred + 1e-9)) / y_true.shape[0]

def classification_errors(y_true, y_pred):
    return np.sum(np.argmax(y_true, axis=1) != np.argmax(y_pred, axis=1))

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, eta=0.01, epochs=10, batch_size=32, momentum=0.9):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.eta = eta
        self.epochs = epochs
        self.batch_size = batch_size
        self.momentum = momentum
        
        self.weights_input = np.random.randn(input_size, hidden_size) * 0.01
        self.bias_input = np.zeros((1, hidden_size))
        self.weights_hidden = np.random.randn(hidden_size, output_size) * 0.01
        self.bias_hidden = np.zeros((1, output_size))

        self.velocity_weights_input = np.zeros_like(self.weights_input)
        self.velocity_bias_input = np.zeros_like(self.bias_input)
        self.velocity_weights_hidden = np.zeros_like(self.weights_hidden)
        self.velocity_bias_hidden = np.zeros_like(self.bias_hidden)

    def forward(self, x):
        self.hidden_input = np.dot(x, self.weights_input) + self.bias_input
        self.hidden_output = np.tanh(self.hidden_input) 
        self.output_input = np.dot(self.hidden_output, self.weights_hidden) + self.bias_hidden
        self.output = softmax(self.output_input)  
        return self.output

    def backward(self, x, y_true):
        output_error = self.output - y_true
        hidden_error = np.dot(output_error, self.weights_hidden.T) * (1 - np.tanh(self.hidden_input) ** 2)

        grad_weights_hidden = np.dot(self.hidden_output.T, output_error) / x.shape[0]
        grad_bias_hidden = np.sum(output_error, axis=0, keepdims=True) / x.shape[0]
        grad_weights_input = np.dot(x.T, hidden_error) / x.shape[0]
        grad_bias_input = np.sum(hidden_error, axis=0, keepdims=True) / x.shape[0]

        self.velocity_weights_hidden = self.momentum * self.velocity_weights_hidden - self.eta * grad_weights_hidden
        self.velocity_bias_hidden = self.momentum * self.velocity_bias_hidden - self.eta * grad_bias_hidden
        self.velocity_weights_input = self.momentum * self.velocity_weights_input - self.eta * grad_weights_input
        self.velocity_bias_input = self.momentum * self.velocity_bias_input - self.eta * grad_bias_input

        self.weights_hidden += self.velocity_weights_hidden
        self.bias_hidden += self.velocity_bias_hidden
        self.weights_input += self.velocity_weights_input
        self.bias_input += self.velocity_bias_input

    def train(self, train_images, train_labels, test_images, test_labels, epochs=10, batch_size=32):
        train_errors = []
        test_errors = []
        train_losses = []
        test_losses = []

        for epoch in range(epochs):
            epoch_train_errors = 0
            epoch_train_loss = 0

            for i in range(0, train_images.shape[0], batch_size):
                batch_images = train_images[i:i+batch_size]
                batch_labels = train_labels[i:i+batch_size]

                one_hot_labels = np.zeros((batch_labels.shape[0], 10))
                one_hot_labels[np.arange(batch_labels.shape[0]), batch_labels] = 1

                output = self.forward(batch_images)

                loss = cross_entropy_loss(one_hot_labels, output)
                epoch_train_loss += loss

                errors = classification_errors(one_hot_labels, output)
                epoch_train_errors += errors

                self.backward(batch_images, one_hot_labels)

            train_errors.append(epoch_train_errors)
            train_losses.append(epoch_train_loss)

            test_loss, test_error = self.evaluate(test_images, test_labels)
            test_errors.append(test_error)
            test_losses.append(test_loss)
            
            if epoch % 5 == 0:
                print(f'Epoch {epoch}/{epochs}, Train Loss: {epoch_train_loss}, Train Errors: {epoch_train_errors}, Test Loss: {test_loss}, Test Errors: {test_error}')

        return train_errors, test_errors, train_losses, test_losses

    def evaluate(self, images, labels):
        total_loss = 0
        total_errors = 0
        for i in range(0, images.shape[0], 32):
            batch_images = images[i:i+32]
            batch_labels = labels[i:i+32]

            one_hot_labels = np.zeros((batch_labels.shape[0], 10))
            one_hot_labels[np.arange(batch_labels.shape[0]), batch_labels] = 1

            output = self.forward(batch_images)

            loss = cross_entropy_loss(one_hot_labels, output)
            total_loss += loss

            errors = classification_errors(one_hot_labels, output)
            total_errors += errors

        return total_loss, total_errors

    def test(self, test_images, test_labels):
        correct_predictions = 0
        for i in range(test_images.shape[0]):
            output = self.forward(test_images[i:i+1])
            predicted_label = np.argmax(output)
            if predicted_label == test_labels[i]:
                correct_predictions += 1
        accuracy = correct_predictions / test_images.shape[0]
        print(f'Test Accuracy: {accuracy * 100:.2f}%')
        return accuracy

plt.ion()
np.random.seed(29)

train_images = load_mnist_images('train-images-idx3-ubyte.gz')
train_labels = load_mnist_labels('train-labels-idx1-ubyte.gz')
test_images = load_mnist_images('t10k-images-idx3-ubyte.gz')
test_labels = load_mnist_labels('t10k-labels-idx1-ubyte.gz')

nn = NeuralNetwork(input_size=784, hidden_size=100, output_size=10, eta=0.01, momentum=0.9)
train_errors, test_errors, train_losses, test_losses = nn.train(train_images, train_labels, test_images, test_labels, epochs=30, batch_size=16)

plt.figure()
plt.plot(train_errors, label='Train Errors')
plt.plot(test_errors, label='Test Errors')
plt.xlabel('Epochs')
plt.ylabel('Classification Errors')
plt.title('Epochs vs Classification Errors')
plt.legend()

plt.figure()
plt.plot(train_losses, label='Train Loss')
plt.plot(test_losses, label='Test Loss')
plt.xlabel('Epochs')
plt.ylabel('Energy (Loss)')
plt.title('Epochs vs Energy (Loss)')
plt.legend()

nn.test(test_images, test_labels)

plt.ioff()
plt.show()