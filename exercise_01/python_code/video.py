import cv2
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import time

from pose_vector_to_transformation_matrix import pose_vector_to_transformation_matrix
from project_points import project_points
from undistort_image import undistort_image
from undistort_image_vectorized import undistort_image_vectorized


def main():
    poses = np.loadtxt("./data/poses.txt")
    K = np.loadtxt("./data/K.txt")
    D = np.loadtxt("./data/D.txt")

    size = 0.08  # side length of the cube in meters
    offset_x = 0.0  # offset in x direction from the first corner in meters
    offset_y = 0.0  # offset in y direction from the first corner in meters

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

    # Define cube edges for drawing
    edges = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4),
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    ]

    fig, ax = plt.subplots(figsize=(10, 8))

    def animate_frame(frame):
        ax.clear()

        # Load image for current frame (assuming images are numbered starting from 1)
        img_path = f"./data/images/img_{frame+1:04d}.jpg"
        try:
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"Warning: Could not load image {img_path}")
                return

            # Undistort the image
            img_undistorted = undistort_image_vectorized(img, K, D)

            # Get pose for current frame
            if frame < len(poses):
                T_C_W = pose_vector_to_transformation_matrix(poses[frame, :])

                # Transform cube points to camera coordinates
                cube_pts_cam_pose = (
                    T_C_W @ np.hstack((cube_pts, np.ones((cube_pts.shape[0], 1)))).T
                )[:3, :]

                # Project to 2D
                cube_pts_2d = project_points(cube_pts_cam_pose, K, D)

                # Display image
                ax.imshow(img_undistorted, cmap="gray")

                # Plot cube corners
                ax.plot(cube_pts_2d[0, :], cube_pts_2d[1, :], "r.", markersize=8)

                # Draw cube edges
                for start, end in edges:
                    ax.plot(
                        [cube_pts_2d[0, start], cube_pts_2d[0, end]],
                        [cube_pts_2d[1, start], cube_pts_2d[1, end]],
                        "r-",
                        linewidth=2,
                    )

                ax.set_title(f"Frame {frame+1}: Projected Cube")
            else:
                print(f"Warning: No pose data for frame {frame+1}")

        except Exception as e:
            print(f"Error processing frame {frame+1}: {e}")
            return

        ax.axis("off")

    # Determine number of frames (use minimum of poses or estimate from images)
    num_frames = min(len(poses), 700)  # Adjust max frames as needed

    # Create animation
    ani = animation.FuncAnimation(
        fig, animate_frame, frames=num_frames, interval=10, repeat=True
    )

    # Save as video (optional)
    # ani.save('cube_projection.mp4', writer='ffmpeg', fps=10)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
