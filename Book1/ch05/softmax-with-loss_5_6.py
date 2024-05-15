import numpy as np

class SoftmaxWithLoss:
  def __init__(self):
    self.loss = None  # 손실
    self.y = None     # softmax의 출력
    self.t = None     # 정답 레이블(원-핫 벡터)
  
  def forward(self, x, t):
    self.t = t
    self.y = softmax(x)
    self.loss = cross_entropy_error(self.y, self.t)
    return self.loss

  def backward(self, dout = 1):
    batch_size = self.t.shape[0]
    dx = (self.y - self.t) / batch_size
    return dx



# 소프트맥스 함수 구현
def softmax(a): 
  c = np.max(a) # 오버플로우 막기 위해 최댓값 이용
  exp_a = np.exp(a - c) # -c : 오버플로우 방지
  sum_exp_a = np.sum(exp_a)
  y = exp_a / sum_exp_a

  return y

# 교차 엔트로피 오차
# -('정답 레이블 * log(추정값)'의 합)
def cross_entropy_error(y, t):
  delta = 1e-7  # 로그 함수에 0이 오지 못하도록 매우 작은 값 더함
  return -np.sum(t * np.log(y + delta))