import numpy as np

from distort_points import distort_points


# modified in 3.2
def project_points(points_3d: np.ndarray, K: np.ndarray, D: np.ndarray) -> np.ndarray:
    """
    Projects 3d points to the image plane, given the camera matrix,
    and distortion coefficients.

    Args:
        points_3d: 3d points (3xN)
        K: camera matrix (3x3)
        D: distortion coefficients (4x1)

    Returns:
        projected_points: 2d points (2xN)
    """
    uv = K @ points_3d  # matmult
    uv_norm = uv[:2] / uv[2]  # normalize by z

    uv_distorted = uv_norm
    # uv_distorted = distort_points(uv_norm.T, D, K).T  # 3.2
    return uv_distorted
