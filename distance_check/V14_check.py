import cv2 as cv
from cv2 import aruco
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

# Load calibration data
calib_data_path = "calib_data/calibration_data.npz"
calib_data = np.load(calib_data_path)
cam_mat, dist_coef = calib_data["cameraMatrix"], calib_data["distCoeffs"]

MARKER_SIZE = 6  # centimeters (measure your printed marker size)
marker_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_250)
param_markers = aruco.DetectorParameters()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Initialize marker data as empty initially
marker_data = np.zeros((0, 3))


def update(frame):
    global marker_data

    ret, frame = cap.read()
    if not ret:
        return

    gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    marker_corners, marker_IDs, _ = aruco.detectMarkers(
        gray_frame, marker_dict, parameters=param_markers)

    if marker_corners:
        rVecs, tVecs, _ = aruco.estimatePoseSingleMarkers(
            marker_corners, MARKER_SIZE, cam_mat, dist_coef)

        # Collect marker positions
        marker_positions = np.array([tVec[0] for tVec in tVecs])

        # Update marker data
        marker_data = np.vstack((marker_data, marker_positions))

        # Clear the previous plot
        ax.clear()

        # Plot camera
        ax.scatter(0, 0, 0, c='red', marker='s', label='Camera')

        # Plot markers
        ax.scatter(marker_data[:, 0], marker_data[:, 1],
                   marker_data[:, 2], c='blue', marker='o', label='Markers')

        # Set plot labels and limits
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.legend()

    return ax


# Set up video capture
cap = cv.VideoCapture(0)

ani = FuncAnimation(fig, update, interval=50)
plt.show()


cap.release()
cv.destroyAllWindows()
