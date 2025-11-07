import numpy as np


def estimatePoseDLT(p, P, K):
    # Estimates the pose of a camera using a set of 2D-3D correspondences
    # and a given camera matrix.
    #
    # p  [n x 2] array containing the undistorted coordinates of the 2D points
    # P  [n x 3] array containing the 3D point positions
    # K  [3 x 3] camera matrix
    #
    # Returns a [3 x 4] projection matrix of the form
    #           M_tilde = [R_tilde | alpha * t]
    # where R is a rotation matrix. M_tilde encodes the transformation
    # that maps points from the world frame to the camera frame

    # Convert 2D to normalized coordinates
    # TODO: Your code here

    p = np.hstack((p, np.ones((len(p), 1)))).T
    K_inv = np.linalg.inv(K)

    norm_coords = np.matmul(K_inv, p).T
    norm_coords = np.divide(norm_coords, norm_coords[:, 2].reshape((-1, 1)))[
        :, :2
    ].reshape((-1, 1))
    # Build measurement matrix Q
    # TODO: Your code here
    Q_1 = np.hstack((P, np.ones((len(P), 1)), np.zeros((len(P), 4)))).reshape(-1, 4)
    Q_2 = np.hstack((np.zeros((len(P), 4)), P, np.ones((len(P), 1)))).reshape(-1, 4)
    Q_3 = -1 * (Q_1 + Q_2) * norm_coords

    Q = np.hstack((Q_1, Q_2, Q_3))
    # Solve for Q.M_tilde = 0 subject to the constraint ||M_tilde||=1
    # TODO: Your code here

    U, S, Vh = np.linalg.svd(Q)
    m_tilde = Vh.T[:, -1]
    # Extract [R | t] with the correct scale
    # TODO: Your code here
    Rt = m_tilde.reshape((3, 4))

    # Find the closest orthogonal matrix to R
    # https://en.wikipedia.org/wiki/Orthogonal_Procrustes_problem
    # TODO: Your code here

    U, S, Vt = np.linalg.svd(Rt[:3, :3])
    R_tilde = U @ Vt

    # Normalization scheme using the Frobenius norm:
    # recover the unknown scale using the fact that R_tilde is a true rotation matrix
    # TODO: Your code here

    scale = np.sqrt(3) / np.linalg.norm(Rt[:3, :3], "fro")

    # Build M_tilde with the corrected rotation and scale
    # TODO: Your code here

    m_tilde = m_tilde * scale
    Rt_corrected = m_tilde.reshape((3, 4))

    return Rt_corrected
