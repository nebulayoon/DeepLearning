# 이 프로젝트에서는 앞으로의 신경망 공부에서 사용될 여러 계층들을 구현할 예정임
import numpy as np

class Relu:
  def __init__(self):
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

# sigmoid function의 computational graph를 보면 forward propagation output만으로 backward propagation을 구할 수 있다.

class Sigmoid:
  def __init__(self):
    self.out = None
  
  def forward(self, x):
    out = 1 / (1 + np.exp(-x))
    self.out = out

    return out

  def backword(self, dout): # y에 대한 L의 미분 y(1-y)
    dx = dout * (1.0 - self.out) * self.out

    return dx

class Affine: # 추가적인 이해필요
  def __init__(self, W, b):
    self.W = W
    self.b = b
    self.x = None
    self.dW = None
    self.db = None

  def forward(self, x):
    self.x = x
    out = np.dot(x, self.W) + self.b
    
    return out

  def backward(self, dout):
    dx = np.dot(dout, self.W.T)
    self.dW = np.dot(self.x.T, dout)
    self.db = np.sum(dout, axis=0)

    return dx