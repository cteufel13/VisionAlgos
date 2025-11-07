import cv2
import matplotlib.pyplot as plt
import numpy as np
import time

from pose_vector_to_transformation_matrix import pose_vector_to_transformation_matrix
from project_points import project_points
from undistort_image import undistort_image
from undistort_image_vectorized import undistort_image_vectorized


def main():

    # load camera poses

    # each row i of matrix 'poses' contains the transformations that transforms
    # points expressed in the world frame to
    # points expressed in the camera frame

    poses = np.loadtxt("./data/poses.txt")

    # define 3D corner positions
    # [Nx3] matrix containing the corners of the checkerboard as 3D points
    # (X,Y,Z), expressed in the world coordinate system

    square_size = 0.04

    x_positions = np.arange(0, 9, 1) * square_size
    y_positions = np.arange(0, 6, 1) * square_size

    corner_matrix = np.meshgrid(x_positions, y_positions)
    corners_3d = np.vstack(
        (
            corner_matrix[0].flatten(),
            corner_matrix[1].flatten(),
            np.zeros(corner_matrix[0].size),
        )
    ).T
    # in meters

    # load camera intrinsics
    K = np.loadtxt("./data/K.txt")
    D = np.loadtxt("./data/D.txt")

    # load one image with a given index

    img = cv2.imread("./data/images/img_0001.jpg", cv2.IMREAD_GRAYSCALE)
    # project the corners on the image
    # compute the 4x4 homogeneous transformation matrix that maps points
    # from the world to the camera coordinate frame

    T_C_W = pose_vector_to_transformation_matrix(poses[0, :])

    # transform 3d points from world to current camera pose
    # TODO: Your code here

    corners_3d_cam_pose = (
        T_C_W @ np.hstack((corners_3d, np.ones((corners_3d.shape[0], 1)))).T
    )[:3, :]

    # project the 3d points to the image plane

    corners_2d = project_points(corners_3d_cam_pose, K, D)

    # plot the image and the projected points

    """fig = plt.figure()
    plt.imshow(img_undistorted, cmap="gray")
    plt.plot(corners_2d[0, :], corners_2d[1, :], "r.", markersize=10)
    plt.axis("off")
    plt.title("Projected corners")
    plt.show()
    plt.close(fig)"""

    # 2.3 Make Cube:

    size = 0.08  # side length of the cube in meters
    offset_x = 0.0  # offset in x direction from the first corner in meters
    offset_y = 0.0  # offset in y direction from the first corner in

    cube_pts = np.array(
        [
            [0, 0, 0],
            [0, size, 0],
            [size, size, 0],
            [size, 0, 0],
            [0, 0, -size],
            [0, size, -size],
            [size, size, -size],
            [size, 0, -size],
        ]
    )  # Generic Coords of Cube with Size

    cube_pts += np.array([offset_x, offset_y, 0])  # Offset of origin

    cube_pts_cam_pose = (
        T_C_W @ np.hstack((cube_pts, np.ones((cube_pts.shape[0], 1)))).T
    )[:3, :]

    cube_pts_2d = project_points(cube_pts_cam_pose, K, D)

    img_undistorted = undistort_image(img, K, D, bilinear_interpolation=True)

    # Plotting Points on the Board + Cube

    """fig2 = plt.figure()
    plt.imshow(img_undistorted, cmap="gray")
    plt.plot(corners_2d[0, :], corners_2d[1, :], "r.", markersize=10)
    # fmt: off
    edges = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]
    # fmt: on
    for start, end in edges:
        plt.plot(
            [cube_pts_2d[0, start], cube_pts_2d[0, end]],
            [cube_pts_2d[1, start], cube_pts_2d[1, end]],
            "r-",
            linewidth=2,
        )
    plt.axis("off")
    plt.title("Projected corners and cube")
    plt.show()"""


if __name__ == "__main__":
    main()
