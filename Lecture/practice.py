import numpy as np
import matplotlib.pyplot as plt

class MLP:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.1):
        # Initialize the model architecture
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate

        # Initialize weights and biases for the hidden and output layers
        self.weights_input_hidden = []
        self.bias_hidden = []
        
        for i in range(len(hidden_size)):
            if i == 0:
                self.weights_input_hidden.append(np.random.randn(self.input_size, hidden_size[i]))
            else:
                self.weights_input_hidden.append(np.random.randn(hidden_size[i-1], hidden_size[i]))
            self.bias_hidden.append(np.zeros((1, hidden_size[i])))
        
        self.weights_hidden_output = np.random.randn(hidden_size[-1], self.output_size)
        self.bias_output = np.zeros((1, self.output_size))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def forward(self, x):
        # Compute the hidden layer outputs
        self.hidden_outputs = []
        self.hidden_inputs = []
        
        for i in range(len(self.hidden_size)):
            if i == 0:
                self.hidden_inputs.append(np.dot(x, self.weights_input_hidden[i]) + self.bias_hidden[i])
            else:
                self.hidden_inputs.append(np.dot(self.hidden_outputs[i-1], self.weights_input_hidden[i]) + self.bias_hidden[i])
            self.hidden_outputs.append(self.sigmoid(self.hidden_inputs[i]))

        # Compute the output layer output
        self.output_input = np.dot(self.hidden_outputs[-1], self.weights_hidden_output) + self.bias_output
        self.output = self.sigmoid(self.output_input)

        return self.output

    def backward(self, x, y, output):
        # Compute the loss
        error = y - output

        # Backpropagate the error to update the weights and biases
        delta_output = error * self.sigmoid_derivative(output)
        delta_hidden = []
        
        for i in range(len(self.hidden_size) - 1, -1, -1):
            if i == len(self.hidden_size) - 1:
                delta_hidden.append(delta_output.dot(self.weights_hidden_output.T) * self.sigmoid_derivative(self.hidden_outputs[i]))
            else:
                delta_hidden.append(delta_hidden[-1].dot(self.weights_input_hidden[i+1].T) * self.sigmoid_derivative(self.hidden_outputs[i]))

        # Update weights and biases
        self.weights_hidden_output += self.hidden_outputs[-1].T.dot(delta_output) * self.learning_rate
        self.bias_output += np.sum(delta_output, axis=0, keepdims=True) * self.learning_rate
        
        for i in range(len(self.hidden_size)):
            if i == 0:
                self.weights_input_hidden[i] += x.T.dot(delta_hidden[-1-i]) * self.learning_rate
            else:
                self.weights_input_hidden[i] += self.hidden_outputs[i-1].T.dot(delta_hidden[-1-i]) * self.learning_rate
            self.bias_hidden[i] += np.sum(delta_hidden[-1-i], axis=0, keepdims=True) * self.learning_rate

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
    mlp = MLP(input_size=2, hidden_size=[4, 4], output_size=1, learning_rate=0.1)
    error_history = mlp.train(inputs, xor_target.reshape(-1, 1), epochs=10000)

    # Plot the error history
    plt.plot(error_history)
    plt.xlabel('Epochs')
    plt.ylabel('Error')
    plt.title('Training Error Over Time')
    plt.show()
