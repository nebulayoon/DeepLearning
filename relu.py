import numpy as np

def relu(x):
    return np.maximum(0, x) # 더 큰놈 출력

A = np.array([1, 2, 3, 4])
print(A)

print(np.ndim(A))
print(A.shape)
print(A.shape[0])

B = np.array([[1, 2], [3, 4], [5, 6]])
print(np.ndim(B))
print(B.shape)