import numpy as np
import matplotlib.pyplot as plt
from itertools import product
import math


#사용되는 activation function
def step(x):
    return 1 if x >= 0.5 else 0


def sigmoid(x):
    return 1 / (1 + np.exp(-x))
def g_sigmoid(x):
    return (1 - x) * x

def generate_a(x, w):
    z = np.dot(x, w)
    a = sigmoid(z)
    return a

#사용되는 gate
def AND_gate(inputs):
    return all(inputs)
def OR_gate(inputs):
    return any(inputs)
def XOR_gate(inputs):
    return sum(inputs) % 2

losses = []

class MLP:
    def __init__(self) -> None:
        #switch = int(input("gate input 선택: '1'입력 or dount input 선택 '-1'입력: "))
        switch = -1

        #generate (dount or gate) input and target
        if switch == -1:  
            self.X = [[0, 0], [0, 1], [1, 0], [1, 1], [0.5,1], [1,0.5], [0,0.5], [0.5,0], [0.5,0.5]]
            self.X = np.insert(self.X, 0, np.array(np.ones(1)), axis=1)
            self.T = [[0], [0], [0], [0], [0], [0], [0], [0], [1]]
            self.input_size = 2
            self.target_size = 9
        else:
            #gate_n = int(input("gate를 선택 and=1, or=2, xor=3: "))
            gate_n = 3
            #self.input_size = int(input("input_size를 입력(self.input_size > 1): "))
            self.input_size = 2
            #target_size와 target만들기 (target=input_combination)
            self.target_size = 2**self.input_size
            input_combiantion = list(product([0., 1.], repeat=self.input_size))
            
            self.X = np.array(input_combiantion)  # 입력
            self.X = np.insert(self.X, 0, np.array(np.ones(1)), axis=1)
            
            self.T = []                           # 출력
            if(gate_n == 1):
                for inputs in input_combiantion:
                    self.T.append([int(AND_gate(inputs))])
            elif(gate_n == 2):
                for inputs in input_combiantion:
                    self.T.append([int(OR_gate(inputs))])
            elif(gate_n == 3):
                for inputs in input_combiantion:
                    self.T.append([int(XOR_gate(inputs))])
            self.T = np.array(self.T)
        
        #self.layer_n = int(input("self.layer_size를 입력: "))
        #self.layer_n = 2
        
        #self.hidden_size = []
        # for i in range(self.layer_n):
        #     self.hidden_size.append(int(input("hidden_layer의 각 size를 입력: ")))
        self.hidden_size = [4, 4]
        self.layer_n = len(self.hidden_size) + 1

        #fixed output_size
        self.output_size = 1
        
        # xavier weight initialization
        #받는 list node 갯수 만큼 만들고 입력 list node 갯수만큼 append
        
        self.w = []
        """
        for i in range (0, self.layer_n):
            self.w.append([])

        for _ in range(0, self.input_size + 1):
            self.w[0].append(np.random.normal(0 ,math.sqrt(1/(self.input_size + 1 + self.hidden_size[0])), self.hidden_size[0]))
        for n in range(1, self.layer_n - 1):
            for i in range(0, self.hidden_size[n] + 1):
                self.w[n].append(np.random.normal(0 ,math.sqrt(1/(self.input_size + 1 + self.hidden_size[n])), self.hidden_size[n]))
        for _ in range(0, self.hidden_size[-1] + 1):
            self.w[self.layer_n - 1].append(np.random.normal(0 ,math.sqrt(1/(self.input_size + 1 + self.output_size)), self.output_size))

        for i in range(self.layer_n):
            self.w[i] = np.array(self.w[i])
        """
        for n in range(self.layer_n): 
            if n == 0:
                self.w.append(np.random.randn(self.input_size + 1, self.hidden_size[0]).tolist())
            elif n < self.layer_n - 1:
                self.w.append(np.random.randn(self.hidden_size[n-1] + 1, self.hidden_size[n]).tolist())
            else:
                self.w.append(np.random.randn(self.hidden_size[n-1] + 1, self.output_size).tolist())
            
        for _ in range(len(self.w)):
            self.w[_] = np.array(self.w[_])
            
        
        #layer_size=2, hidden_size=[2,2] 일때, and의 경우:0.5, or의 경우: 0.1, xor의 경우: 0.015
        self.learning_rate = 0.
        
        self.loss = 0
        self.losses = []
        
        #bias 초기화
        self.b = []
        for i in range (0, self.layer_n - 1):
            self.b.append(np.ones(self.target_size))
        #행의 개수(=다음 노드의 갯수)에 맞게 1로 초기화 
        

    def forward_propagation(self):
        self.a = []
        self.back_a = []
        
        for i in range(self.layer_n):
            if i == 0:
                self.a.append(generate_a(self.X, self.w[i]))
                self.back_a.append(generate_a(self.X, self.w[i]))
                self.a[i] = np.insert(self.a[i], 0, self.b[i], axis=1)
            elif i < (self.layer_n - 1):
                self.a.append(generate_a(self.a[i-1], self.w[i]))
                self.back_a.append(generate_a(self.a[i-1], self.w[i]))
                self.a[i] = np.insert(self.a[i], 0, self.b[i], axis=1)
            else :
                self.a.append(generate_a(self.a[i-1], self.w[i]))
                self.back_a.append(generate_a(self.a[i-1], self.w[i]))
        
        Y = self.a[-1].reshape(self.target_size, 1)
        
        self.loss = np.sum(((Y - self.T) ** 2)) 
        self.losses.append(self.loss)
        self.g_loss = Y - self.T
        
        return Y
        
    def back_propagation(self):
        g_z = []
        g_w = []
        g_b = []
            
        i = self.layer_n
        while i > 0:
            if i == (self.layer_n):
                tmp = self.g_loss * g_sigmoid(self.back_a[i-1])
                g_w.append(self.a[i-1].T @ tmp)
                #
                g_b.append((tmp @ self.w[i-1].T).T[0]) #bias 배열
                g_z.append(np.delete(tmp @ self.w[i-1].T, 0, axis=1))
            elif i > 1:
                tmp = g_z[-1] * g_sigmoid(self.back_a[i-1])
                g_w.append(self.a[i-1].T @ tmp)
                #
                g_b.append((tmp @ self.w[i-1].T).T[0]) #bias 배열
                g_z.append(np.delete(tmp @ self.w[i-1].T, 0, axis=1))
            elif i == 1:
                tmp = g_z[-1] * g_sigmoid(self.back_a[i-1])
                g_w.append(self.X.T @ tmp)
                #
                self.X.T[0] -= self.learning_rate * np.sum((tmp @ self.w[i-1].T).T[0]) #bias 배열
            i -= 1
        
        g_w.reverse()
        g_b.reverse()

        #update rule
        #self.w = self.w-self.learning_rate*(1/n)*(X.T(Z-T))
        for i in range(self.layer_n):
            self.w[i] -= self.learning_rate * (1/(self.target_size)) * g_w[i]
            #self.w[i] -= self.learning_rate  * g_w[i]
        for i in range(self.layer_n - 1):
            #print(np.sum(g_b[i]))
            self.b[i] -= self.learning_rate * np.sum(g_b[i])
        
    def train(self):
        max_epochs = 100000
        
        for i in range(max_epochs):
            Y = self.forward_propagation()
            #반올림
            # Y = np.array([step(_) for _ in Y])
            # Y = Y.reshape(self.target_size, 1)
            """
            반올림 말고, 가능 오차범위 적용하기 
            """
            L = self.loss
            print(L)
            if L < 0.01:
                print(f'iter {i}, Loss: {np.sum(L)}, self.learning_rate: {self.learning_rate}')
                print('Y:', Y)
                break
            if i == max_epochs -1:
                print(Y)
            self.back_propagation()
    
    def plot_loss(self):
        self.train()
        plt.plot(self.losses)
        plt.title(f"Learning Process for Gate")
        plt.xlabel("Iteration")
        plt.ylabel("loss Rate")
        plt.show()

if __name__ == '__main__':
    m = MLP()
    m.plot_loss()