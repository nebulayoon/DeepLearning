import numpy as np

a = np.array([0.3, 2.9, 4.0])
exp_a = np.exp(a)
print(exp_a)

sum_exp_a = np.sum(exp_a)
print(sum_exp_a)

y = exp_a / sum_exp_a
print(y)


def softmax(a): # 이 식은 오버플로우의 문제가 있음. 지수 함수라는 것은 큰 값을 내는 함수이기에..
    exp_a = np.exp(a)
    sum_exp_a = np.sum(exp_a)
    y = exp_a /sum_exp_a

    return y

a = np.array([1010, 1000, 990])
c = np.max(a)

print(np.exp(a - c) / np.sum(np.exp(a - c)))

def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a

    return y

a = np.array([0.3, 2.9, 4.0])
y = softmax(a)
print(y)
print(np.sum(y))