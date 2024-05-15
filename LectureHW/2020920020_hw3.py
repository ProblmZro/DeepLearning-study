import numpy as np
import matplotlib.pyplot as plt

class MLP:
  # Initialize the values
  # Suppose input_size = 2, output_size = 1, learning_rate = 0.1
  def __init__(self, hidden_size, input_size = 2, output_size = 1, learning_rate = 0.1):
    self.learning_rate = learning_rate
    self.input_size = input_size
    self.hidden_size = hidden_size
    self.output_size = output_size

    self.weights_input_hidden = []
    self.bias_hidden = []

    # Initialize random weights and bias
    for i in range(len(hidden_size)):
      for i in range(len(hidden_size)):
        if i == 0:
          self.weights_input_hidden.append(np.random.randn(self.input_size, hidden_size[i]))
        else:
          self.weights_input_hidden.append(np.random.randn(hidden_size[i-1], hidden_size[i]))
        self.bias_hidden.append(np.zeros((1, hidden_size[i])))

    self.weights_hidden_output = np.random.randn(self.hidden_size[-1], self.output_size)
    self.bias_output = np.random.randn(1, self.output_size)

  # Calculate the sigmoid function (active function)
  def sigmoid(self, x):
    return 1 / (1 + np.exp(-x))
  
  # Calculate the derivation of sigmoid function (active function)
  def sigmoid_derivation(self, x):
    return (1 - x) * x 

  # Implement the forward pass
  def forward(self, x):
    self.hidden_outputs = []
    self.hidden_inputs = []
        
    for i in range(len(self.hidden_size)):
      if i == 0:
        # Calculate the inputs to the first hidden layer
        self.hidden_inputs.append(np.dot(x, self.weights_input_hidden[i]) + self.bias_hidden[i])
      else:
        # Calculate the inputs to subsequent hidden layers
        self.hidden_inputs.append(np.dot(self.hidden_outputs[i-1], self.weights_input_hidden[i]) + self.bias_hidden[i])
        # Apply the sigmoid activation function to the hidden layer outputs
      self.hidden_outputs.append(self.sigmoid(self.hidden_inputs[i]))

    # Compute the input to the output layer
    self.output_input = np.dot(self.hidden_inputs[-1], self.weights_hidden_output) + self.bias_output
    # Apply the sigmoid activation function to the output layer
    self.output = self.sigmoid(self.output_input)

    return self.output

 # Implement the backward pass
  def backward(self, x, y, output):
    # Calculate the error between the target and actual output
    error = y - output
    # Calculate the delta for the output layer
    delta_output = error * self.sigmoid_derivation(output)

    delta_hidden = []
    for i in range(len(self.hidden_size) - 1, -1, -1):
      if i == len(self.hidden_size) - 1:
        # Calculate the delta for the last hidden layer
        delta_hidden.append(delta_output.dot(self.weights_hidden_output.T) * self.sigmoid_derivation(self.hidden_outputs[i]))
      else:
        # Calculate the delta for previous hidden layers
        delta_hidden.append(delta_hidden[-1].dot(self.weights_input_hidden[i+1].T) * self.sigmoid_derivation(self.hidden_outputs[i]))

    # Update the weights and biases
    self.weights_hidden_output += self.hidden_outputs[-1].T.dot(delta_output) * self.learning_rate
    self.bias_output += np.sum(delta_output, axis=0, keepdims=True) * self.learning_rate

    for i in range(len(self.hidden_size)):
      if i == 0:
        # Update the weights and biases for the first hidden layer
        self.weights_input_hidden[i] += x.T.dot(delta_hidden[-1-i]) * self.learning_rate
      else:
         # Update the weights and biases for subsequent hidden layers
        self.weights_input_hidden[i] += self.hidden_outputs[i-1].T.dot(delta_hidden[-1-i]) * self.learning_rate
      self.bias_hidden[i] += np.sum(delta_hidden[-1-i], axis=0, keepdims=True) * self.learning_rate

  # Train the perceptron and get error rates
  def training(self, x, y, max_epochs=50000, target_error=0.01):
     # To store the error rate for each epoch
    error_history = []
    # Iterate for a maximum epochs
    for _ in range(max_epochs):
      # Perform a forward pass to find the actual output
      output = self.forward(x)
      # Perform a backward pass
      self.backward(x, y, output)

      # Calculate the current error rate and store
      error = np.mean(np.abs(y - output))
      # Check the current error is below the target error threshold
      error_history.append(error)
      if error < target_error:
        break

    return error_history

  def plt_learning(self, input, target, gate_type):
    # Train the perceptron and get error rates(result)
    res = self.training(input, target.reshape(-1, 1))

    # Plot the error rate during training
    plt.plot(res)
    plt.title(f"Learning Process for {gate_type} Gate")
    plt.xlabel("Iteration")
    plt.ylabel("Error Rate(%)")
    plt.show()

    # Plot the decision boundary during training
    # 1. Setting the boundary of x, y axis
    x_min, x_max = -0.2, 1.2
    y_min, y_max = -0.2, 1.2
    # 2. Making grid point
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 400), np.linspace(y_min, y_max, 400))
    # 3. Flatten input points into a two-dimensional array
    input_data = np.c_[xx.ravel(), yy.ravel()]
    # 4. Predict the output and transform to grid shape
    predictions = self.forward(input_data).reshape(xx.shape)

    # 5. Drawing contour graph
    plt.contourf(xx, yy, predictions, levels=[0, 0.5, 1], cmap=plt.cm.RdBu, alpha=0.8)
    plt.scatter(input[:, 0], input[:, 1], c=target, cmap=plt.cm.RdBu_r, s=100)
    plt.title(f'Decision Boundary for {gate_type} Gate')
    plt.show()


if __name__ == '__main__':
    # Define input data and target values for each logic gates
    gate_input = np.array([(0, 0), (0, 1), (1, 0), (1, 1)])
    and_target = np.array([0, 0, 0, 1])
    or_target = np.array([0, 1, 1, 1])
    xor_target = np.array([0, 1, 1, 0])
    # Define donut data and target values
    donut_input = np.array([(0.,0.), (0.,1.), (1.,0.), (1.,1.), (0.5,1.), (1.,0.5), (0.,0.5), (0.5,0.), (0.5,0.5)])
    donut_target = np.array([0,0,0,0,0,0,0,0,1])

    # Setting the number of layers and nodes
    # (Suppose 2 level layers and each layer have 4 nodes)
    ''' if) hidden_size=[3, 4, 3, 2]
        Number of nodes in the first hidden layer = 3
        Number of nodes in the second hidden layer = 4
        Number of nodes in the third hidden layer = 3
        Number of nodes in the first hidden layer = 2
    '''
    hidden_size=[4,4]

    # Implement MLP for each logic gates
    and_mlp = MLP(hidden_size)
    or_mlp = MLP(hidden_size)
    xor_mlp = MLP(hidden_size)
    donut_mlp = MLP(hidden_size)

    # Visualize the learning process for each training
    and_mlp.plt_learning(gate_input, and_target, "AND")
    or_mlp.plt_learning(gate_input, or_target, "OR")
    xor_mlp.plt_learning(gate_input, xor_target, "XOR")
    donut_mlp.plt_learning(donut_input, donut_target, "DONUT")