import numpy as np


def decomposeEssentialMatrix(E):
    """Given an essential matrix, compute the camera motion, i.e.,  R and T such
    that E ~ T_x R

    Input:
      - E(3,3) : Essential matrix

    Output:
      - R(3,3,2) : the two possible rotations
      - u3(3,1)   : a vector with the translation information
    """
    U, S, Vt = np.linalg.svd(E)
    W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])

    R1 = U @ W @ Vt
    R2 = U @ W.T @ Vt

    if np.linalg.det(R1) < 0:
        R1 *= -1
    elif np.linalg.det(R2) < 0:
        R2 *= -1

    R = np.zeros((3, 3, 2))
    R[:, :, 0] = R1
    R[:, :, 1] = R2

    # Translation (up to sign) is the last column of U
    u3 = U[:, 2].reshape(3, 1)

    return R, u3
