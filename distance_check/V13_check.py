import cv2 as cv
from cv2 import aruco
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load calibration data
calib_data_path = "calib_data/calibration_data.npz"
calib_data = np.load(calib_data_path)
cam_mat = calib_data["cameraMatrix"]
dist_coef = calib_data["distCoeffs"]
r_vectors = calib_data["rvecs"]
t_vectors = calib_data["tvecs"]

MARKER_SIZE = 5  # centimeters (measure your printed marker size)
marker_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_250)
param_markers = aruco.DetectorParameters()


def plot_markers_and_camera_in_3d(x_vals, y_vals, z_vals, camera_pos, marker_ids):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x_vals, y_vals, z_vals, marker='o', color='red')

    # Plot camera position as a black dot marker with the label 'Camera'
    ax.scatter(camera_pos[0], camera_pos[1], camera_pos[2],
               marker='o', color='black', s=100, label='Camera')

    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')
    ax.set_title('Markers and Camera in 3D')
    ax.legend()

    for i, txt in enumerate(marker_ids):
        ax.text(x_vals[i], y_vals[i], z_vals[i],
                f"ID: {txt}\n({x_vals[i]}, {y_vals[i]}, {z_vals[i]})")

    plt.show()


cap = cv.VideoCapture(0)

# Replace these with the actual camera coordinates if known
camera_x, camera_y, camera_z = 0.0, 0.0, 0.0  # Example camera coordinates
camera_position = [camera_x, camera_y, camera_z]

plot_active = False
x_values, y_values, z_values = [], [], []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    height, width = frame.shape[:2]
    cv.line(frame, (int(width / 2) - 10, int(height / 2)),
            (int(width / 2) + 10, int(height / 2)), (255, 0, 0), 2)
    cv.line(frame, (int(width / 2), int(height / 2) - 10),
            (int(width / 2), int(height / 2) + 10), (255, 0, 0), 2)
    cv.putText(frame, "0,0", (int(width / 2) + 5, int(height / 2) - 5),
               cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    marker_corners, marker_IDs, _ = aruco.detectMarkers(
        gray_frame, marker_dict, parameters=param_markers
    )

    if marker_corners:
        rVec, tVec, _ = aruco.estimatePoseSingleMarkers(
            marker_corners, MARKER_SIZE, cam_mat, dist_coef
        )
        total_markers = range(0, marker_IDs.size)

        for ids, corners, i in zip(marker_IDs, marker_corners, total_markers):
            corners = corners.reshape(4, 2)
            corners = corners.astype(int)

            distance = np.sqrt(
                tVec[i][0][2] ** 2 + tVec[i][0][0] ** 2 + tVec[i][0][1] ** 2
            )

            cv.drawFrameAxes(
                frame, cam_mat, dist_coef, rVec[i], tVec[i], MARKER_SIZE * 0.6
            )

        if key == ord('s'):
            if not plot_active:
                captured_frame = frame.copy()
                x_values = [round(tVec[i][0][0], 1)
                            for i in range(len(marker_IDs))]
                y_values = [round(tVec[i][0][1], 1)
                            for i in range(len(marker_IDs))]
                z_values = [round(tVec[i][0][2], 1)
                            for i in range(len(marker_IDs))]

                # Pass marker positions (x, y, z), camera position, and marker IDs for display
                plot_markers_and_camera_in_3d(
                    x_values, y_values, z_values, camera_position, marker_IDs.flatten())
                plot_active = True
            else:
                plt.close()
                x_values, y_values, z_values = [], [], []
                plot_active = False

        if key == ord("q"):
            break

    cv.imshow("frame", frame)
    key = cv.waitKey(1)

cap.release()
cv.destroyAllWindows()
