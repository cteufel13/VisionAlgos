import numpy as np
from scipy import signal


def harris(img, patch_size, kappa):
    """Returns the harris scores for an image given a patch size and a kappa value
    The returned scores are of the same shape as the input image"""

    x_filter = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    y_filter = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    Ix = signal.convolve2d(img, x_filter, mode="valid")
    Iy = signal.convolve2d(img, y_filter, mode="valid")
    Ix2 = Ix**2
    Iy2 = Iy**2
    Ixy = Ix * Iy

    summation_filter = np.ones((patch_size, patch_size))

    sum_x2 = signal.convolve2d(Ix2, summation_filter, mode="valid")
    sum_y2 = signal.convolve2d(Iy2, summation_filter, mode="valid")
    sum_xy = signal.convolve2d(Ixy, summation_filter, mode="valid")

    R_pre = sum_x2 * sum_y2 - sum_xy**2 - kappa * (sum_x2 + sum_y2) ** 2
    R_post = np.pad(R_pre, 1 + patch_size)

    return R_post
