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

    self.weights_input_hidden = []
    self.bias_hidden = []

    for i in range(len(hidden_size)):
      for i in range(len(hidden_size)):
        if i == 0:
          self.weights_input_hidden.append(np.random.randn(self.input_size, hidden_size[i]))
        else:
          self.weights_input_hidden.append(np.random.randn(hidden_size[i-1], hidden_size[i]))
        self.bias_hidden.append(np.zeros((1, hidden_size[i])))

    self.weights_hidden_output = np.random.randn(self.hidden_size[-1], self.output_size)
    self.bias_output = np.random.randn(1, self.output_size)

  def sigmoid(self, x):
    return 1 / (1 + np.exp(-x))
  
  def sigmoid_derivation(self, x):
    return (1 - x) * x 

  def forward(self, x):
    self.hidden_outputs = []
    self.hidden_inputs = []
        
    for i in range(len(self.hidden_size)):
      if i == 0:
        self.hidden_inputs.append(np.dot(x, self.weights_input_hidden[i]) + self.bias_hidden[i])
      else:
        self.hidden_inputs.append(np.dot(self.hidden_outputs[i-1], self.weights_input_hidden[i]) + self.bias_hidden[i])
      self.hidden_outputs.append(self.sigmoid(self.hidden_inputs[i]))

    # Compute the output layer output
    self.output_input = np.dot(self.hidden_inputs[-1], self.weights_hidden_output) + self.bias_output
    self.output = self.sigmoid(self.output_input)

    return self.output

  def backward(self, x, y, output):
    error = y - output
    delta_output = error * self.sigmoid_derivation(output)

    delta_hidden = []
    for i in range(len(self.hidden_size) - 1, -1, -1):
      if i == len(self.hidden_size) - 1:
        delta_hidden.append(delta_output.dot(self.weights_hidden_output.T) * self.sigmoid_derivation(self.hidden_outputs[i]))
      else:
        delta_hidden.append(delta_hidden[-1].dot(self.weights_input_hidden[i+1].T) * self.sigmoid_derivation(self.hidden_outputs[i]))

    self.weights_hidden_output += self.hidden_outputs[-1].T.dot(delta_output) * self.learning_rate
    self.bias_output += np.sum(delta_output, axis=0, keepdims=True) * self.learning_rate

    for i in range(len(self.hidden_size)):
      if i == 0:
        self.weights_input_hidden[i] += x.T.dot(delta_hidden[-1-i]) * self.learning_rate
      else:
        self.weights_input_hidden[i] += self.hidden_outputs[i-1].T.dot(delta_hidden[-1-i]) * self.learning_rate
      self.bias_hidden[i] += np.sum(delta_hidden[-1-i], axis=0, keepdims=True) * self.learning_rate

  # x : inputs, y : targets
  def training(self, x, y, max_epochs=10000):
    error_history = []
    for _ in range(max_epochs):
      output = self.forward(x)

      self.backward(x, y, output)

      error_history.append(np.mean(np.abs(y - output)))

    return error_history

  def plt_learning(self, target, gate_type):
    res = self.training(inputs, target.reshape(-1, 1))
    plt.plot(res)
    plt.title(f"Learning Process for {gate_type} Gate")
    plt.xlabel("Iteration")
    plt.ylabel("Error Rate(%)")
    plt.show()

    x_min, x_max = -0.2, 1.2
    y_min, y_max = -0.2, 1.2
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 400), np.linspace(y_min, y_max, 400))
    input_data = np.c_[xx.ravel(), yy.ravel()]
    predictions = self.forward(input_data).reshape(xx.shape)

    plt.contourf(xx, yy, predictions, levels=[0, 0.5, 1], cmap=plt.cm.RdBu, alpha=0.8)
    plt.scatter(inputs[:, 0], inputs[:, 1], c=target, cmap=plt.cm.RdBu_r, s=100)
    plt.xlabel('Input 1')
    plt.ylabel('Input 2')
    plt.title('Decision Boundary')
    plt.show()


if __name__ == '__main__':
    inputs = np.array([(0, 0), (0, 1), (1, 0), (1, 1)])
    and_target = np.array([0, 0, 0, 1])
    or_target = np.array([0, 1, 1, 1])
    xor_target = np.array([0, 1, 1, 0])

    hidden_size=[4,4]
    and_mlp = MLP(hidden_size)
    or_mlp = MLP(hidden_size)
    xor_mlp = MLP(hidden_size)

    and_mlp.plt_learning(and_target, "AND")
    or_mlp.plt_learning(or_target, "OR")
    xor_mlp.plt_learning(xor_target, "XOR")