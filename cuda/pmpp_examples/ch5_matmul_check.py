import numpy as np


def print_result(left_dim, mid_dim, right_dim):
    A = np.zeros(left_dim * mid_dim)

    for j in range(left_dim * mid_dim):
        A[j] = 0.2 * (-5 + (j % 10))

    B = np.zeros(mid_dim * right_dim)

    for j in range(mid_dim * right_dim):  
      B[j] = 0.1 * (0.1 + (j % 8))

    C = np.matmul(A.reshape(left_dim, mid_dim), B.reshape(mid_dim, right_dim))

    for j in range(3):
        print(C.reshape(-1)[j])
    
    print("...")
    for j in range(left_dim * right_dim - 3, left_dim * right_dim):
        print(C.reshape(-1)[j])



if __name__ == "__main__":
    left_dim = 1024
    mid_dim = 512
    right_dim = 256

    left_dim = 1024 * 4
    mid_dim = 1024 * 4
    right_dim = 1024 * 4
    print_result(left_dim, mid_dim, right_dim)