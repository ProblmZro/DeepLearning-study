import numpy as np

class Relu:
  def __init__(self):
    # mask : T, F로 구성된 넘파이 배열
    # 순전파 입력인 x가 0 이하이면 T
    self.mask = None
  
  def forward(self, x):
    self.mask = (x <= 0)
    out = x.copy()
    out[self.mask] = 0
    return out

  def backward(self, dout):
    dout[self.mask] = 0
    dx = dout
    return dx

x = np.array([[1.0, -0.5], [-2.0, 3.0]])
print(x)
mask = (x <= 0)
print(mask)