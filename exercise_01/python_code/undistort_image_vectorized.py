import numpy as np

from distort_points import distort_points


# 3.3
def undistort_image_vectorized(
    img: np.ndarray, K: np.ndarray, D: np.ndarray
) -> np.ndarray:
    """
    Undistorts an image using the camera matrix and distortion coefficients.

    Args:
        img: distorted image (HxW)
        K: camera matrix (3x3)
        D: distortion coefficients (4x1)

    Returns:
        und_img: undistorted image (HxW)
    """
    height, width = img.shape[:2]

    # Create meshgrid (MATLAB uses 1-based indexing, Python uses 0-based)
    X, Y = np.meshgrid(range(width), range(height))

    # Flatten and create coordinate array (transpose to match MATLAB's column format)
    px_locs = np.column_stack([X.ravel(), Y.ravel()]).T

    # Get distorted coordinates
    dist_px_locs = distort_points(px_locs.T, D, K).T

    # Round coordinates
    dist_x = np.round(dist_px_locs[0, :]).astype(int)
    dist_y = np.round(dist_px_locs[1, :]).astype(int)

    # Create mask for valid coordinates
    valid_mask = (dist_x >= 0) & (dist_x < width) & (dist_y >= 0) & (dist_y < height)

    # Initialize output image
    undimg = np.zeros(height * width, dtype=img.dtype)

    # Vectorized indexing (equivalent to MATLAB's linear indexing)
    valid_indices = dist_y[valid_mask] * width + dist_x[valid_mask]
    undimg[valid_mask] = img.flat[valid_indices]

    # Reshape back to original image shape
    undimg = undimg.reshape((height, width))

    return undimg
