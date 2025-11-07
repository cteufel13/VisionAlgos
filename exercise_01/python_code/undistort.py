import cv2
import matplotlib.pyplot as plt
import numpy as np
import time

from pose_vector_to_transformation_matrix import pose_vector_to_transformation_matrix
from project_points import project_points
from undistort_image import undistort_image
from undistort_image_vectorized import undistort_image_vectorized

import os


def main():

    K = np.loadtxt("./data/K.txt")
    D = np.loadtxt("./data/D.txt")

    images_distorted = os.listdir("./data/images/")
    for image in images_distorted:

        if image.endswith(".jpg"):
            img = cv2.imread(
                os.path.join("./data/images/", image), cv2.IMREAD_GRAYSCALE
            )
            img_undistorted = undistort_image_vectorized(img, K, D)
            cv2.imwrite(
                os.path.join("./data/images_undistorted/", image), img_undistorted
            )


if __name__ == "__main__":
    main()
