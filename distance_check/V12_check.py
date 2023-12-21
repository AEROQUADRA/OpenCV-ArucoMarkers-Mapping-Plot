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


def plot_markers_and_relative_on_graph(x_vals, y_vals, relative_x_vals, relative_y_vals, marker_ids):
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.scatter(x_vals, y_vals, marker='o', color='red')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('Markers Plotted on Graph')
    for i, txt in enumerate(marker_ids):
        plt.annotate(f"ID: {txt}\n({x_vals[i]}, {y_vals[i]})",
                     (x_vals[i], y_vals[i]), textcoords="offset points", xytext=(5, 5), ha='center')
        print(f"Marker ID: {txt}, X: {x_vals[i]}, Y: {y_vals[i]}")
    plt.grid()

    plt.subplot(1, 2, 2)
    plt.scatter(np.round(relative_x_vals), np.round(relative_y_vals),
                marker='o', color='blue')  # Rounding to whole numbers
    plt.xlabel('Relative X-axis')
    plt.ylabel('Relative Y-axis')
    plt.title('Relative Markers Plotted on Graph')
    for i, txt in enumerate(marker_ids):
        plt.annotate(f"ID: {txt}\n({round(relative_x_vals[i])}, {round(relative_y_vals[i])})",  # Rounding to whole numbers
                     (round(relative_x_vals[i]), round(relative_y_vals[i])), textcoords="offset points", xytext=(5, 5), ha='center')
        print(
            f"Relative Marker ID: {txt}, Relative X: {relative_x_vals[i]}, Relative Y: {relative_y_vals[i]}")
    plt.grid()

    plt.tight_layout()
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

            cv.drawFrameAxes(
                frame, cam_mat, dist_coef, rVec[i], tVec[i], MARKER_SIZE * 0.6
            )

            cv.putText(frame, f"id: {ids[0]} Dist: {round(distance, 2)}",
                       top_right,
                       cv.FONT_HERSHEY_PLAIN,
                       1.3,
                       (0, 0, 255),
                       2,
                       cv.LINE_AA,
                       )
            cv.putText(frame, f"x:{round(tVec[i][0][0],1)} y: {round(tVec[i][0][1],1)} ",
                       bottom_right,
                       cv.FONT_HERSHEY_PLAIN,
                       1.0,
                       (0, 0, 255),
                       2,
                       cv.LINE_AA,
                       )

            if ids[0] != marker_IDs[0][0]:  # Ignore closest marker itself
                x_relative = tVec[i][0][0] - tVec[0][0][0]
                y_relative = tVec[i][0][1] - tVec[0][0][1]

                bottom_left = corners[3].ravel()
                cv.putText(frame, f"x_rel:{round(x_relative,1)} y_rel:{round(y_relative,1)} ",
                           bottom_left,
                           cv.FONT_HERSHEY_PLAIN,
                           1.0,
                           (255, 200, 0),
                           2,
                           cv.LINE_AA,
                           )

        min_distance = np.inf
        closest_marker_ID = None
        for ids, corners, i in zip(marker_IDs, marker_corners, total_markers):
            corners = corners.reshape(4, 2)
            bottom_left = corners[3].ravel()

            distance_to_bottom_left = np.linalg.norm(
                bottom_left - np.array([0, frame.shape[0]])
            )

            if i == 0 or distance_to_bottom_left < min_distance:
                min_distance = distance_to_bottom_left
                closest_marker_ID = ids[0]

        if closest_marker_ID:
            closest_marker_index = np.where(
                marker_IDs == closest_marker_ID)[0][0]
            closest_marker_corners = marker_corners[closest_marker_index]
            cv.polylines(
                frame,
                [closest_marker_corners.astype(np.int32)],
                True,
                (0, 255, 0),
                4,
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

            for i in range(len(marker_IDs)):
                marker_id = marker_IDs[i][0]
                distance = np.sqrt(
                    tVec[i][0][2] ** 2 + tVec[i][0][0] ** 2 + tVec[i][0][1] ** 2)
                print(
                    f"Marker ID: {marker_id}, Coordinates (x, y): ({x_values[i]}, {y_values[i]}), Distance: {round(distance, 2)}")

            relative_x_values = [round(tVec[i][0][0] - tVec[0][0][0], 1)
                                 for i in range(len(marker_IDs))]
            relative_y_values = [round(tVec[i][0][1] - tVec[0][0][1], 1)
                                 for i in range(len(marker_IDs))]

            plot_markers_and_relative_on_graph(
                x_values, y_values, relative_x_values, relative_y_values, marker_IDs.flatten())
            plot_active = True
        else:
            plt.close()
            x_values, y_values = [], []
            plot_active = False

    if key == ord("q"):
        break

cap.release()
cv.destroyAllWindows()
