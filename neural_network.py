import numpy as np

def sigmoid(a1):
    return 1 / (1 + np.exp(a1))

def identity_function(x): #항등함수
    return x

X = np.array([1, 2])

W = np.array([[1, 3, 5], [2, 4, 6]])

Y = np.dot(X, W)
print(Y)

x = np.array([1.0, 0.5])
w1 = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
b1 = np.array([0.1, 0.2, 0.3])

a1 = np.dot(x, w1) + b1
print(a1)
z1 = sigmoid(a1)
print(z1)

w2 = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
b2 = np.array([0.1, 0.2])

a2 = np.dot(z1, w2) + b2
z2 = sigmoid(a2)

print(a2)
print(z2)

w3 = np.array([[0.1, 0.3], [0.2, 0.4]])
b3 = np.array([0.1, 0.2])

a3 = np.dot(z2, w3) + b3
y = identity_function(a3)
print(y)