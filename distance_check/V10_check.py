import cv2 as cv
from cv2 import aruco
import numpy as np
import matplotlib.pyplot as plt


print("0")

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


print("1")

print("plot_markers_on_graph start")


def plot_markers_on_graph(x_vals, y_vals, marker_ids):
    plt.figure()
    plt.scatter(x_vals, y_vals, marker='o', color='red')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('Markers Plotted on Graph')
    for i, txt in enumerate(marker_ids):
        plt.annotate(f"ID: {txt}\n({x_vals[i]}, {y_vals[i]})",
                     (x_vals[i], y_vals[i]), textcoords="offset points", xytext=(5, 5), ha='center')
        print(f"Marker ID: {txt}, X: {x_vals[i]}, Y: {y_vals[i]}")
    plt.grid()
    plt.show()
    print("plot_markers_on_graph inside end")


print("plot_markers_on_graph end")

cap = cv.VideoCapture(0)

plot_active = False
x_values, y_values = [], []

print("video cam started")

while True:
    print("while loop started")
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
            cv.putText(frame, f"x:{round(tVec[i][0][0], 1)} y: {round(tVec[i][0][1], 1)} ",
                       bottom_right,
                       cv.FONT_HERSHEY_PLAIN,
                       1.0,
                       (0, 0, 255),
                       2,
                       cv.LINE_AA,
                       )

            rMat, _ = cv.Rodrigues(rVec[i])
            angles = cv.RQDecomp3x3(rMat)[0]
            cv.putText(frame, f"Roll: {round(angles[0], 2)} Pitch: {round(angles[1], 2)} Yaw: {round(angles[2], 2)}",
                       (bottom_right[0], bottom_right[1] + 30),
                       cv.FONT_HERSHEY_PLAIN,
                       1.0,
                       (0, 0, 255),
                       2,
                       cv.LINE_AA,
                       )

        # Integrate the logic to detect and highlight marker in the bottom left corner
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

            # Print marker information
            for i in range(len(marker_IDs)):
                marker_id = marker_IDs[i][0]
                distance = np.sqrt(
                    tVec[i][0][2] ** 2 +
                    tVec[i][0][0] ** 2 + tVec[i][0][1] ** 2
                )
                rMat, _ = cv.Rodrigues(rVec[i])
                angles = cv.RQDecomp3x3(rMat)[0]
                print(
                    f"Marker ID: {marker_id}, Coordinates (x, y): ({x_values[i]}, {y_values[i]}), Distance: {round(distance, 2)}")
                print(
                    f"Rotation Information - Roll: {round(angles[0], 2)}, Pitch: {round(angles[1], 2)}, Yaw: {round(angles[2], 2)}")

            # Pass marker IDs for display
            plot_markers_on_graph(x_values, y_values, marker_IDs.flatten())
            plot_active = True
        else:
            plt.close()
            x_values, y_values = [], []
            plot_active = False

    if key == ord("q"):
        break

cap.release()
cv.destroyAllWindows()
