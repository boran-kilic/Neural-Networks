import numpy as np  
import matplotlib.pyplot as plt 

np.random.seed(57)
plt.ion()

n = 300
x = np.random.uniform(0, 1, (n, 1))
v = np.random.uniform(-0.1, 0.1, (n, 1))
d = np.sin(20*x) + 3*x + v

plt.figure()
plt.scatter(x, d, color='blue', s=10)
plt.xlabel('x')
plt.ylabel('d')
plt.title('Data')    

def tanh_derivative(x):
    return 1 - np.tanh(x) ** 2

class NeuralNetwork:
    def __init__(self, N=24, eta=0.01):
        self.N = N
        self.eta = eta
        self.weights_input = np.random.randn(N, 1)
        self.bias_input = np.random.randn(N, 1)
        self.weights_hidden = np.random.randn(1, N)
        self.bias_hidden = np.random.randn(1, 1)
    
    def forward(self, x):
        self.hidden_input = np.dot(self.weights_input, x.T) + self.bias_input
        self.hidden_output = np.tanh(self.hidden_input)
        self.output = np.dot(self.weights_hidden, self.hidden_output) + self.bias_hidden
        return self.output.T
    
    def train(self, x, d, epochs=1000):
        mse_list = []
        for epoch in range(epochs):
            mse = 0
            for i in range(n):
                xi = x[i].reshape(1, 1)
                di = d[i].reshape(1, 1)
                
                hidden_input = np.dot(self.weights_input, xi) + self.bias_input
                hidden_output = np.tanh(hidden_input)
                output = np.dot(self.weights_hidden, hidden_output) + self.bias_hidden
                
                error = di - output
                mse += error ** 2
                
                delta_output = error
                delta_hidden = tanh_derivative(hidden_input) * (self.weights_hidden.T * delta_output)
                
                self.weights_hidden += self.eta * delta_output * hidden_output.T
                self.bias_hidden += self.eta * delta_output
                self.weights_input += self.eta * delta_hidden * xi.T
                self.bias_input += self.eta * delta_hidden
            
            mse /= n
            mse_list.append(mse.item())
            if epoch > 1 and mse_list[-1] > mse_list[-2]:
                self.eta *= 0.9
            
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, MSE: {mse.item():.6f}")
        
        return mse_list

nn = NeuralNetwork()
mse_values = nn.train(x, d, epochs=3000)

plt.figure()
plt.plot(mse_values, label='MSE')
plt.xlabel('Epochs')
plt.ylabel('MSE')
plt.title('Training Error')
plt.legend()

x_fit = np.linspace(0, 1, 100).reshape(-1, 1)
y_fit = nn.forward(x_fit)

plt.figure()
plt.scatter(x, d, color='blue', s=10, label='Data')
plt.plot(x_fit, y_fit, color='red', linewidth=2, label='Fitted Curve')
plt.xlabel('x')
plt.ylabel('d')
plt.title('Curve Fitting with Neural Network')
plt.legend()

plt.ioff()
plt.show()
