import numpy as np
import matplotlib.pyplot as plt

class Perceptron:
  # Initialize random weights
  def __init__(self):
    self.weights = np.random.rand(3)

  # Define the step function
  def activation(self, x):
    return 1 if x >= 0 else 0

  # Calculate the weighted sum & apply activation function
  def prediction(self, x):
    weights_sum = np.dot(x, self.weights)
    return self.activation(weights_sum)

  # Calculate the decision boundary (line equation)
  def decision_boundary(self, x):
    return (-self.weights[1] * x - self.weights[0]) / self.weights[2]

  # Drawing the decision boundary line in graph
  def plot_decision_boundary(self, gate_type):
    # Define points for each logic gates to plot
    points = \
      {'AND': [(0, 0, 'or'), (0, 1, 'or'), (1, 0, 'or'), (1, 1, 'ob')], 
      'OR': [(0, 0, 'or'), (0, 1, 'ob'), (1, 0, 'ob'), (1, 1, 'ob')], 
      'XOR': [(0, 0, 'or'), (0, 1, 'ob'), (1, 0, 'ob'), (1, 1, 'or')]}
    
    # Plot the points and decision boundary line
    for x, y, marker in points[gate_type]:
      plt.plot(x, y, marker)
    plt.ylim([-0.5, 1.5])
    plt.xlabel('x-axis')
    plt.ylabel('y-axis')
    plt.title(f'Decision boundary for {gate_type} Gate')
    plt.legend()
    plt.show()

  def learning(self, inputs, targets, learning_rate=0.1, max_epochs=1000):
    error_percentage = [] # To store the error rate for each epoch
    for i in range(max_epochs):
      print("----------------")
      print(f"w0, w1, w2: {self.weights[0]}, {self.weights[1]}, {self.weights[2]}")
      
      # Plot the decision boundary line (the boundary of x value is 0 ~ 1)
      x = np.linspace(0, 1, 100)
      plt.plot(x, self.decision_boundary(x), label=i)
      
      errors = 0  # To count the number of misclassifications in this epoch
      for j in range(len(inputs)):
        real_inputs = np.concatenate(([1], inputs[j])) # Insert the bias value 1 in inputs
        target = targets[j]
        output = self.prediction(real_inputs)
        error = target - output 
        self.weights += learning_rate * error * real_inputs # Update the weights
        errors += int(error != 0) # Add errors when misclassifications is occured

      error_percentage.append(errors / len(inputs) * 100) # Calculate error rate for this epoch

      # If there are no errors, stop training early
      if errors == 0: break 
          
    return error_percentage

  def plt_learning(self, inputs, target, gate_type):
    # Train the perceptron and get error rates
    error_percentage = self.learning(inputs, target)

    # Plot the decision boundary during training
    self.plot_decision_boundary(gate_type)

    # Plot the error rate during training
    plt.title(f"Learning Process for {gate_type} Gate")
    plt.xlabel("Iteration")
    plt.ylabel("Error Rate(%)")
    print(error_percentage)
    plt.plot(error_percentage, 'or')
    plt.show()

if __name__ == '__main__':
  # Define input data and target values for each logic gates
  inputs = np.array([(0, 0), (0, 1), (1, 0), (1, 1)])
  and_target = np.array([x[0] and x[1] for x in inputs])
  or_target = np.array([x[0] or x[1] for x in inputs])
  xor_target = np.array([0, 1, 1, 0])

  # Create perceptrons for different logic gates and train the perceptrons
  and_perceptron = Perceptron()
  or_perceptron = Perceptron()
  xor_perceptron = Perceptron()

  # Visualize the learning process for each logic gate
  and_perceptron.plt_learning(inputs, and_target, "AND")
  or_perceptron.plt_learning(inputs, or_target, "OR")
  xor_perceptron.plt_learning(inputs, xor_target, "XOR")