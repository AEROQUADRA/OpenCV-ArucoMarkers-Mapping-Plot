import cv2 as cv
from cv2 import aruco
import numpy as np

# Update the path to 'calibration_data.npz' in check.py
calib_data_path = "calib_data/calibration_data.npz"

calib_data = np.load(calib_data_path)
print(calib_data.files)  # Confirm the exact keys in the loaded file

# Access the data using the correct keys
cam_mat = calib_data["cameraMatrix"]
dist_coef = calib_data["distCoeffs"]
r_vectors = calib_data["rvecs"]
t_vectors = calib_data["tvecs"]

MARKER_SIZE = 6  # centimeters (measure your printed marker size)

marker_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_250)

param_markers = aruco.DetectorParameters()

cap = cv.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    marker_corners, marker_IDs, reject = aruco.detectMarkers(
        gray_frame, marker_dict, parameters=param_markers
    )
    if marker_corners:
        min_distance = np.inf  # Initialize minimum distance to a large value
        closest_marker_ID = None  # Initialize ID of the closest marker
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
            top_left = corners[1].ravel()
            bottom_right = corners[2].ravel()
            bottom_left = corners[3].ravel()

            # Calculate the distance from the bottom left corner of the frame
            # Bottom left corner coordinates
            bottom_left_frame = np.array([0, frame.shape[0]])
            distance_to_bottom_left = np.linalg.norm(
                bottom_left_frame - bottom_left)

            # Determine the closest marker to the bottom left corner
            if i == 0 or distance_to_bottom_left < min_distance:
                min_distance = distance_to_bottom_left
                closest_marker_ID = ids[0]  # ID of the closest marker

        # Highlight the closest marker
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
    if key == ord("q"):
        break

cap.release()
cv.destroyAllWindows()
