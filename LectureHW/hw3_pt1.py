# 은닉층 1개일 때

import numpy as np
import matplotlib.pyplot as plt

class MLP:
  # Initialize random weights
  def __init__(self, hidden_size, input_size = 2, output_size = 1, learning_rate = 0.1):
    self.learning_rate = learning_rate
    self.input_size = input_size
    self.hidden_size = hidden_size
    self.output_size = output_size

    self.weights_input_hidden = np.random.rand(self.input_size, self.hidden_size)
    self.weights_hidden_output = np.random.rand(self.hidden_size, self.output_size)
    self.bias_hidden = np.random.randn(1, self.hidden_size)
    self.bias_output = np.random.randn(1, self.output_size)

  def sigmoid(self, x):
    return 1 / (1 + np.exp(-x))
  
  def sigmoid_derivation(self, x):
    return (1 - x) * x 

  def forward(self, x):
    # Compute the hidden layer output
    self.hidden_input = np.dot(x, self.weights_input_hidden) + self.bias_hidden
    self.hidden_output = self.sigmoid(self.hidden_input)

    # Compute the output layer output
    self.output_input = np.dot(self.hidden_output, self.weights_hidden_output) + self.bias_output
    self.output = self.sigmoid(self.output_input)

    return self.output

  def backward(self, x, y, output):
    error = y - output
    delta_output = error * self.sigmoid_derivation(output)
    delta_hidden =  delta_output.dot(self.weights_hidden_output.T) * self.sigmoid_derivation(self.hidden_output)

    self.weights_hidden_output += self.hidden_output.T.dot(delta_output) * self.learning_rate
    self.bias_output += np.sum(delta_output, axis=0, keepdims=True) * self.learning_rate
    self.weights_input_hidden += x.T.dot(delta_hidden) * self.learning_rate
    self.bias_hidden += np.sum(delta_hidden, axis=0, keepdims=True) * self.learning_rate

  # x : inputs, y : targets
  def training(self, x, y, max_epochs=10000):
    error_history = []
    for _ in range(max_epochs):
      output = self.forward(x)

      self.backward(x, y, output)

      error_history.append(np.mean(np.abs(y - output)))

    return error_history

if __name__ == '__main__':
    inputs = np.array([(0, 0), (0, 1), (1, 0), (1, 1)])
    xor_target = np.array([0, 1, 1, 0])

    mlp = MLP(hidden_size=4)
    res = mlp.training(inputs, xor_target.reshape(-1, 1))

    plt.plot(res)
    plt.xlabel('Epochs')
    plt.ylabel('Error')
    plt.title('Training Error Over Time')
    plt.show()