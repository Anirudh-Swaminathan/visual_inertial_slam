#!/usr/bin/python

# Created by anicodebreaker on March 07, 2020
import timeit
from scipy.linalg import expm
import numpy as np


def main():
    setup = """
import numpy as np
from scipy.linalg import expm
A = np.array([[0, -1, -1], [1, 0, -1], [1, 1, 0]])
    """

    ani_setup = """
import numpy as np
from scipy.linalg import expm
v = np.array([1, -1, 1])
def skew_symm_3D(vec_3d):
    # row 1
    ret_mat = np.zeros((3, 3))
    ret_mat[0][1] = -1.0 * vec_3d[2]
    ret_mat[0][2] = vec_3d[1]

    # row 2
    ret_mat[1][0] = vec_3d[2]
    ret_mat[1][2] = -1.0 * vec_3d[0]

    # row 3
    ret_mat[2][0] = -1.0 * vec_3d[1]
    ret_mat[2][1] = vec_3d[0]
    return ret_mat
    """

    expm_test = """
ans = expm(A)
    """
    print("Timing scipy expm")
    print(timeit.timeit(expm_test, setup, number=2000))

    ani_test = """
A = skew_symm_3D(v)
f1 = np.sin(np.sqrt(3)) / np.sqrt(3)
Asq = np.matmul(A, A)
f2 = (1 - np.cos(np.sqrt(3))) / 3
I = np.identity(3)
ans1 = I + f1*A + f2*Asq
    """
    print("Timing Rodrigues")
    print(timeit.timeit(ani_test, ani_setup, number=2000))


if __name__ == "__main__":
    main()
