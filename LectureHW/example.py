import numpy as np
import matplotlib.pyplot as plt
#mlp = MLP(input_size=2, hidden_size=4, output_size=1, learning_rate=0.1)
class MLP:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.1):
        # Initialize the model architecture
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate

        # Initialize weights and biases for the hidden and output layers
        self.weights_input_hidden = np.random.randn(self.input_size, self.hidden_size)
        self.bias_hidden = np.random.randn(1, self.hidden_size)
        self.weights_hidden_output = np.random.randn(self.hidden_size, self.output_size)
        self.bias_output = np.random.randn(1, self.output_size)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def forward(self, x):
        # Compute the hidden layer output
        self.hidden_input = np.dot(x, self.weights_input_hidden) + self.bias_hidden
        self.hidden_output = self.sigmoid(self.hidden_input)

        # Compute the output layer output
        self.output_input = np.dot(self.hidden_output, self.weights_hidden_output) + self.bias_output
        self.output = self.sigmoid(self.output_input)

        return self.output

    def backward(self, x, y, output):
        # Compute the loss
        error = y - output

        # Backpropagate the error to update the weights and biases
        delta_output = error * self.sigmoid_derivative(output)
        error_hidden = delta_output.dot(self.weights_hidden_output.T)
        delta_hidden = error_hidden * self.sigmoid_derivative(self.hidden_output)

        # Update weights and biases
        self.weights_hidden_output += self.hidden_output.T.dot(delta_output) * self.learning_rate
        self.bias_output += np.sum(delta_output, axis=0, keepdims=True) * self.learning_rate
        self.weights_input_hidden += x.T.dot(delta_hidden) * self.learning_rate
        self.bias_hidden += np.sum(delta_hidden, axis=0, keepdims=True) * self.learning_rate

    def train(self, x, y, epochs):
        error_history = []
        for _ in range(epochs):
            # Forward pass
            output = self.forward(x)

            # Backpropagation and weight updates
            self.backward(x, y, output)

            # Calculate and record the error
            error = np.mean(np.abs(y - output))
            error_history.append(error)

        return error_history

if __name__ == '__main__':
    # Define input data and target values
    inputs = np.array([(0, 0), (0, 1), (1, 0), (1, 1)])
    xor_target = np.array([0, 1, 1, 0])

    # Create and train an MLP
    mlp = MLP(input_size=2, hidden_size=4, output_size=1, learning_rate=0.1)
    error_history = mlp.train(inputs, xor_target.reshape(-1, 1), epochs=10000)

    # Plot the error history
    plt.plot(error_history)
    plt.xlabel('Epochs')
    plt.ylabel('Error')
    plt.title('Training Error Over Time')
    plt.show()
    # Plot the decision boundary
    # x_min, x_max = -0.2, 1.2
    # y_min, y_max = -0.2, 1.2
    # xx, yy = np.meshgrid(np.linspace(x_min, x_max, 400), np.linspace(y_min, y_max, 400))
    # input_data = np.c_[xx.ravel(), yy.ravel()]
    # predictions = mlp.forward(input_data).reshape(xx.shape)

    # plt.contourf(xx, yy, predictions, levels=[0, 0.5, 1], cmap=plt.cm.RdBu, alpha=0.8)
    # plt.scatter(inputs[:, 0], inputs[:, 1], c=xor_target, cmap=plt.cm.RdBu_r, s=100)
    # plt.xlabel('Input 1')
    # plt.ylabel('Input 2')
    # plt.title('Decision Boundary')
    # plt.show()