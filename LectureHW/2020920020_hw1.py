import numpy as np

# four input cases
inputs = [(0, 0), (0, 1), (1, 0), (1, 1)]

# Implementation of AND gate
# a, b: inputs / w1, w2: weight / threshold
def AND(a, b, w1, w2, theta):
    x = np.array([a, b])
    w = np.array([w1, w2])
    
    tmp = np.sum(x * w)
    if tmp < theta : return 0
    else : return 1

if __name__ == '__main__':
    incorrect_cnt = 1

    while incorrect_cnt > 0 :
        print("----------------")
        # Assume w, theta value : 0 ~ 1
        w = np.random.uniform(0, 1, 2)
        theta = np.random.uniform(0, 1)
        print(f"Randomly generated w1, w2: {w[0]}, {w[1]}")
        print(f"Randomly generated threshold theta: {theta}")
        print("\nx1 x2 out")

        incorrect_cnt = 0

        for n in inputs:
            y = AND(n[0], n[1], w[0], w[1], theta)
            print(n[0], n[1], y)

            if y != (n[0] and n[1]):
                incorrect_cnt += 1
    
        if incorrect_cnt > 0:
            print(f"\nNumber of incorrect outputs:", incorrect_cnt)
            print("Retry.")
    
    print(f"\nCorrect!")