import numpy as np

from linear_triangulation import linearTriangulation


def disambiguateRelativePose(Rots, u3, points0_h, points1_h, K1, K2):
    """DISAMBIGUATERELATIVEPOSE- finds the correct relative camera pose (among
    four possible configurations) by returning the one that yields points
    lying in front of the image plane (with positive depth).

    Arguments:
      Rots -  3x3x2: the two possible rotations returned by decomposeEssentialMatrix
      u3   -  a 3x1 vector with the translation information returned by decomposeEssentialMatrix
      p1   -  3xN homogeneous coordinates of point correspondences in image 1
      p2   -  3xN homogeneous coordinates of point correspondences in image 2
      K1   -  3x3 calibration matrix for camera 1
      K2   -  3x3 calibration matrix for camera 2

    Returns:
      R -  3x3 the correct rotation matrix
      T -  3x1 the correct translation vector

      where [R|t] = T_C2_C1 = T_C2_W is a transformation that maps points
      from the world coordinate system (identical to the coordinate system of camera 1)
      to camera 2.
    """
    M1 = K1 @ np.hstack((np.eye(3), np.zeros((3, 1))))

    R1 = Rots[:3, :3, 0]
    R2 = Rots[:3, :3, 1]

    M2_1 = K2 @ np.hstack((R1, u3))
    M2_2 = K2 @ np.hstack((R2, u3))
    P_1 = linearTriangulation(points0_h, points1_h, M1, M2_1)
    P_2 = linearTriangulation(points0_h, points1_h, M1, M2_2)

    c1 = np.sum((P_1[2, :] > 0))
    c2 = np.sum((P_2[2, :] > 0))

    if c1 > c2:
        return R1, u3

    else:
        return R2, u3
