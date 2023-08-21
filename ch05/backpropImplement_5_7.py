import sys, os
sys.path.append(os.pardir)
sys.path.append("/Users/jaymoon/Desktop/coding/deep_learning/")
import numpy as np
from dataset.mnist import load_mnist
from backprop_5_7 import TwoLayerNet

# 데이터 읽기
(x_train, t_train), (x_test, t_test) = \
  load_mnist(normalize=True, one_hot_label=True)
network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

iters_num = 10000 # 반복 횟수
train_size = x_train.shape[0]
batch_size = 100  # 미니배치 크기
learning_rate = 0.1

train_lost_list = []
train_acc_list = []
test_acc_list = []

iter_per_epoch = max(train_size / batch_size, 1)

for i in range(iters_num):
  batch_mask = np.random.choice(train_size, batch_size)
  x_batch = x_train[batch_mask]
  t_batch = t_train[batch_mask]

  ###
  grad = network.gradient(x_batch, t_batch)

  for key in ('W1', 'b1', 'W2', 'b2') :
    network.params[key] -= learning_rate * grad[key]
  
  loss = network.loss(x_batch, t_batch)
  train_lost_list.append(loss)

  if i % iter_per_epoch == 0:
    train_acc = network.accuracy(x_train, t_train)
    test_acc = network.accuracy(x_test, t_test)
    train_acc_list.append(train_acc)
    test_acc_list.append(test_acc)
    print("train acc, test acc | " + str(train_acc) + ", " + str(test_acc))