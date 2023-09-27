import numpy as np
import matplotlib.pyplot as plt

# four input cases
inputs = np.array([(0, 0), (0, 1), (1, 0), (1, 1)])
target = np.array([x[0] and x[1] for x in inputs])

# Implementation of AND gate
# x: inputs / w: weight (x0: bias)
def AND(x1, x2, w0, w1, w2, x0=1):
    x = np.array([x0, x1, x2])
    w = np.array([w0, w1, w2])
    
    tmp = np.sum(x * w)
    if tmp < 0 : return 0
    else : return 1

if __name__ == '__main__':
    # Assume first value of w is 1.0, 2.0, 3.0
    w = np.array([1.0, 2.0, 3.0])
    error_percentage = []

    while True :
        print("----------------")
        print(f"w0, w1, w2: {w[0]}, {w[1]}, {w[2]}")

        output = np.zeros(4)
        for i in range(4):
            x0, x1 = inputs[i]
            output[i] = AND(x0, x1, w[0], w[1], w[2])

        error = output - target
        error_percentage.append(int(sum(error)) / 4 * 100)

        print("output:", output)

        if int(np.sum(error)) == 0:
            print("correct!")
            plt.title(f"Learning Process for And Gate")
            plt.xlabel("Iteration")
            plt.ylabel("Error Rate(%)")
            plt.plot(error_percentage, 'or')
            plt.show()
            break
        
        deltw = np.zeros(3)
        for i in range(4):
            delta = (target[i] - output[i]) * np.array([1, inputs[i][0], inputs[i][1]])
            deltw += 0.5 * delta
        w += deltw
        print("incorrect")