import numpy as np


def reprojectPoints(P, M_tilde, K):
    # Reproject 3D points given a projection matrix
    #
    # P         [n x 3] coordinates of the 3d points in the world frame
    # M_tilde   [3 x 4] projection matrix
    # K         [3 x 3] camera matrix
    #
    # Returns [n x 2] coordinates of the reprojected 2d points
    print(K.shape, M_tilde.shape, P.shape)
    P2 = np.hstack((P, np.ones((len(P), 1)))).T
    points = K @ M_tilde @ P2
    points = points.T
    points = points / points[:, 2].reshape((-1, 1))

    return points[:, :2]
