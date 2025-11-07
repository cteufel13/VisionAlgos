import math
import numpy as np

from distort_points import distort_points


# 3.3
def undistort_image(
    img: np.ndarray, K: np.ndarray, D: np.ndarray, bilinear_interpolation: bool = False
) -> np.ndarray:
    """
    Corrects an image for lens distortion.

    Args:
        img: distorted image (HxW)
        K: camera matrix (3x3)
        D: distortion coefficients (4x1)
        bilinear_interpolation: whether to use bilinear interpolation or not
    """
    undistorted_img = np.zeros_like(img)
    height, width = img.shape[:2]
    x, y = np.meshgrid(range(width), range(height))
    coords = np.column_stack([x.ravel(), y.ravel()])

    distorted_coords = distort_points(coords, D, K)
    distorted_coords_rounded = np.round(distorted_coords).astype(int)  # rounded down

    if not bilinear_interpolation:

        for i in range(width):
            for j in range(height):
                u_d, v_d = distorted_coords_rounded[j * width + i]
                if 0 <= u_d < width and 0 <= v_d < height:
                    undistorted_img[j, i] = img[v_d, u_d]

        return undistorted_img

    else:

        for i in range(width):
            for j in range(height):
                u_d, v_d = distorted_coords[j * width + i]
                u_d0, v_d0 = round(u_d), round(v_d)
                u_d1, v_d1 = math.ceil(u_d), math.ceil(v_d)

                if (
                    0 <= u_d0 < width
                    and 0 <= v_d0 < height
                    and 0 <= u_d1 < width
                    and 0 <= v_d1 < height
                ):
                    du = u_d - u_d0
                    dv = v_d - v_d0

                    # Get the 4 pixel values
                    top_left = img[v_d0, u_d0]
                    top_right = img[v_d0, u_d1]
                    bottom_left = img[v_d1, u_d0]
                    bottom_right = img[v_d1, u_d1]

                    # Bilinear interpolation
                    top = top_left * (1 - du) + top_right * du
                    bottom = bottom_left * (1 - du) + bottom_right * du
                    undistorted_img[j, i] = top * (1 - dv) + bottom * dv

        return undistorted_img
