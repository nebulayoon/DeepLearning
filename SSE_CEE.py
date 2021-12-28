import numpy as np

# y: softmax output
# t: answer table

def sum_of_squares_for_error(y, t):
    return 0.5 * np.sum((y - t)**2)

def cross_entropy_error(y, t):
    delta = 1e-7
    return -np.sum(t * np.log(y + delta))