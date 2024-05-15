import sys, os
sys.path.append(os.pardir)
sys.path.append("/Users/jaymoon/Desktop/coding/deep_learning/")
import numpy as np
from dataset.mnist import load_mnist
from backprop_5_7 import TwoLayerNet

# 기울기 구하는 방법 두가지
# 수치 미분 : 구현 쉬움, 계산 느림
# 오차역전파법 : 구현 어려움, 계산 빠름
# 두 결과를 비교해서 오차역전파법 구현 검증 (기울기 확인)

# 데이터 읽기
(x_train, t_train), (x_test, t_test) = \
  load_mnist(normalize=True, one_hot_label=True)

network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

x_batch = x_train[:3]
t_batch = t_train[:3]

grad_numerical = network.numerical_gradient(x_batch, t_batch)
grad_backprop = network.gradient(x_batch, t_batch)

# 각 가중치의 차이의 절댓값을 구한 후 그 절댓값들의 평균을 냄
for key in grad_numerical.keys():
  diff = np.average(np.abs(grad_backprop[key] - grad_numerical[key]))
  print(key + ":" + str(diff))