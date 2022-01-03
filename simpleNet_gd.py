import numpy as np

def softmax(x):
  if x.ndim == 2:
    x = x.T
    x = x - np.max(x, axis=0)
    y = np.exp(x) / np.sum(np.exp(x), axis=0)
    return y.T 

  x = x - np.max(x) # overflow 대책
  return np.exp(x) / np.sum(np.exp(x))

def cross_entropy_error(y, t):
  if y.ndim == 1:
    t = t.reshape(1, t.size)
    y = y.reshape(1, y.size)
      
  if t.size == y.size:
    t = t.argmax(axis=1)
            
  batch_size = y.shape[0]
  return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size

def numerical_gradient(f, x):
  h = 1e-4 # 0.0001
  grad = np.zeros_like(x)
  
  it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
  while not it.finished:
    idx = it.multi_index
    tmp_val = x[idx]
    x[idx] = float(tmp_val) + h
    fxh1 = f(x) # f(x+h)
    
    x[idx] = tmp_val - h 
    fxh2 = f(x) # f(x-h)
    grad[idx] = (fxh1 - fxh2) / (2*h)
    
    x[idx] = tmp_val # 값 복원
    it.iternext()
      
  return grad

class simpleNet:
  def __init__(self):
    self.W = np.random.randn(2, 3)
  
  def predict(self, x):
    return np.dot(x, self.W)
  
  def loss(self, x, t):
    z = self.predict(x)
    y = softmax(z)
    loss = cross_entropy_error(y, t)

    return loss

net = simpleNet()
print(net.W)

x = np.array([0.6, 0.9])
p = net.predict(x)

print(p)
print(np.argmax(p))

t = np.array([0, 0, 1]) # 정답 레이블
print(net.loss(x, t))

# 기울기 구하기
def f(W):
  return net.loss(x, t)
# f = lambda W: net.loss(x, t)

dW = numerical_gradient(f, net.W)
print(dW)