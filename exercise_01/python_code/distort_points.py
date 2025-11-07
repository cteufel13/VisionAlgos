import numpy as np


def distort_points(x: np.ndarray, D: np.ndarray, K: np.ndarray) -> np.ndarray:
    """
    Applies lens distortion to 2D points xon the image plane.

    Args:
        x: 2d points (Nx2)
        D: distortion coefficients (4x1)
        K: camera matrix (3x3)
    """
    u_0, v_0 = K[0, 2], K[1, 2]
    k_1, k_2 = D[0], D[1]

    r = np.sqrt((x[:, 0] - u_0) ** 2 + (x[:, 1] - v_0) ** 2)

    ud = (1 + k_1 * r**2 + k_2 * r**4) * (x[:, 0] - u_0) + u_0
    vd = (1 + k_1 * r**2 + k_2 * r**4) * (x[:, 1] - v_0) + v_0
    return np.vstack((ud, vd)).T
