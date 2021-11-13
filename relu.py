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

C = np.array([[1,2], [3, 4]])
D = np.array([[5, 6], [7, 8]])
E = np.dot(C, D) # 행렬의 곱
print(E)

print(np.dot(A, [5, 6, 7, 8]))

# 당연하지만, 행렬곱은 A, B와 B, A의 값이 달라질수 있다.

F = np.array([[1, 2, 3], [4, 5, 6]])
G = np.array([[1, 2], [3, 4], [5, 6]])
print(np.dot(F, G))