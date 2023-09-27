import numpy as np
import matplotlib.pyplot as plt

class Perceptron:
    def __init__(self):
        # Initialize weights and bias
        self.weights = np.random.rand(3)
    
    def activation(self, x):
        # Step function as activation
        return 1 if x >= 0 else 0

    def predict(self, x):
        # Calculate the weighted sum
        weighted_sum = np.dot(x, self.weights)
        return self.activation(weighted_sum)
    
    def learn(self, inputs, targets, learning_rate=0.1, max_epochs=1000):
        error_history = []
        
        for epoch in range(max_epochs):
            errors = 0
            for i in range(len(inputs)):
                x = np.concatenate(([1], inputs[i]))  # Add bias input
                target = targets[i]
                prediction = self.predict(x)
                error = target - prediction
                self.weights += learning_rate * error * x
                errors += int(error != 0)
            
            error_rate = errors / len(inputs) * 100
            error_history.append(error_rate)
            
            # Check for convergence
            if errors == 0:
                break
        
        print(error_history)
        return error_history

def plot_learning_process(inputs, targets, perceptron, gate_type):
    error_history = perceptron.learn(inputs, targets)
    print(error_history)
    
    plt.plot(error_history)
    plt.title(f"Learning Process for {gate_type} Gate")
    plt.xlabel("Iteration")
    plt.ylabel("Error Rate(%)")
    plt.show()

if __name__ == '__main__':
    # Define inputs and targets for AND, OR, and XOR gates
    inputs = np.array([(0, 0), (0, 1), (1, 0), (1, 1)])
    and_targets = np.array([0, 0, 0, 1])
    or_targets = np.array([0, 1, 1, 1])
    xor_targets = np.array([0, 1, 1, 0])
    
    # Create perceptrons for each gate
    and_perceptron = Perceptron()
    or_perceptron = Perceptron()
    xor_perceptron = Perceptron()
    
    # Plot learning process and error graphs for each gate
    plot_learning_process(inputs, and_targets, and_perceptron, "AND")
    plot_learning_process(inputs, or_targets, or_perceptron, "OR")
    plot_learning_process(inputs, xor_targets, xor_perceptron, "XOR")
