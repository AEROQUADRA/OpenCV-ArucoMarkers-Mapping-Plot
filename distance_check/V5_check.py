import cv2 as cv
from cv2 import aruco
import numpy as np
import matplotlib.pyplot as plt

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


def plot_markers_on_graph(x_vals, y_vals):
    plt.figure()
    plt.scatter(x_vals, y_vals, marker='o', color='red')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('Markers Plotted on Graph')
    plt.grid()
    plt.show()


cap = cv.VideoCapture(0)

plot_active = False
x_values, y_values = [], []

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
            cv.polylines(
                frame, [corners.astype(np.int32)], True, (0,
                                                          255, 255), 4, cv.LINE_AA
            )
            corners = corners.reshape(4, 2)
            corners = corners.astype(int)
            top_right = corners[0].ravel()
            bottom_right = corners[2].ravel()

            distance = np.sqrt(
                tVec[i][0][2] ** 2 + tVec[i][0][0] ** 2 + tVec[i][0][1] ** 2
            )

            point = cv.drawFrameAxes(
                frame, cam_mat, dist_coef, rVec[i], tVec[i], 4, 4)
            cv.putText(
                frame,
                f"id: {ids[0]} Dist: {round(distance, 2)}",
                top_right,
                cv.FONT_HERSHEY_PLAIN,
                1.3,
                (0, 0, 255),
                2,
                cv.LINE_AA,
            )
            cv.putText(
                frame,
                f"x:{round(tVec[i][0][0],1)} y: {round(tVec[i][0][1],1)} ",
                bottom_right,
                cv.FONT_HERSHEY_PLAIN,
                1.0,
                (0, 0, 255),
                2,
                cv.LINE_AA,
            )

    cv.imshow("frame", frame)
    key = cv.waitKey(1)

    if key == ord('s'):
        if not plot_active:
            captured_frame = frame.copy()
            x_values = [round(tVec[i][0][0], 1)
                        for i in range(len(marker_IDs))]
            y_values = [round(tVec[i][0][1], 1)
                        for i in range(len(marker_IDs))]
            plot_markers_on_graph(x_values, y_values)
            plot_active = True
        else:
            plt.close()
            x_values, y_values = [], []
            plot_active = False

    if key == ord("q"):
        break

cap.release()
cv.destroyAllWindows()
