import numpy as np

# def step_funtion(x):
#     if x > 0:
#         return 1
#     else:
#         return 0

def step_funtion(x):
    y = x > 0
    print(y)
    return y.astype(np.int64)

x = np.array([-1.0, 1.0, 2.0])
print(x)
print(step_funtion(x))