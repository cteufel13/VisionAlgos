import numpy as np


def pose_vector_to_transformation_matrix(pose_vec: np.ndarray) -> np.ndarray:
    """
    Converts a 6x1 pose vector into a 4x4 transformation matrix.

    Args:
        pose_vec: 6x1 vector representing the pose as [wx, wy, wz, tx, ty, tz]

    Returns:
        T: 4x4 transformation matrix
    """
    w = pose_vec[0:3]
    t = pose_vec[3:6]
    theta = np.linalg.norm(w)

    k = pose_vec / theta
    kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])

    R = np.eye(3) + np.sin(theta) * kx + (1 - np.cos(theta)) * (kx @ kx)

    transformation_matrix = np.eye(4)
    transformation_matrix[0:3, 0:3] = R
    transformation_matrix[0:3, 3] = t

    return transformation_matrix
