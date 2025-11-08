import numpy as np


def fundamentalEightPoint(p1, p2):
    """The 8-point algorithm for the estimation of the fundamental matrix F

    The eight-point algorithm for the fundamental matrix with a posteriori
    enforcement of the singularity constraint (det(F)=0).
    Does not include data normalization.

    Reference: "Multiple View Geometry" (Hartley & Zisserman 2000), Sect. 10.1 page 262.

    Input: point correspondences
      - p1 np.ndarray(3,N): homogeneous coordinates of 2-D points in image 1
      - p2 np.ndarray(3,N): homogeneous coordinates of 2-D points in image 2

    Output:
      - F np.ndarray(3,3) : fundamental matrix
    """
    Q = np.zeros((p1.shape[1], 9))
    for i in range(p1.shape[1]):
        kron = np.kron(p1[:, i], p2[:, i])
        kron = np.expand_dims(kron, 1).T
        Q[i] = kron
    U, S, Vt = np.linalg.svd(Q)
    F_vec = Vt[-1, :]

    F = F_vec.reshape(3, 3).T

    U, S, Vt = np.linalg.svd(F)
    S[2] = 0
    S_diag = np.diag(S)

    F = U @ S_diag @ Vt
    return F
