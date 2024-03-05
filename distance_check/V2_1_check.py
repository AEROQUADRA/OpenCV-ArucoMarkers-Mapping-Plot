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

# Load the image you want to process
image_path = 'distance_check/Store/tr1.jpg'  # Replace with your image path
frame = cv.imread(image_path)

gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
marker_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_250)

# Adjust parameters to increase sensitivity
param_markers = aruco.DetectorParameters()
# Default value (not explicitly specified)
param_markers.adaptiveThreshConstant = 7
# Default value (not explicitly specified)
param_markers.minMarkerPerimeterRate = 0.01
# Default value (not explicitly specified)
param_markers.maxMarkerPerimeterRate = 1.0
# Default value (not explicitly specified)
param_markers.polygonalApproxAccuracyRate = 0.05
# Default value (not explicitly specified)
param_markers.minCornerDistanceRate = 0.01
# Default value (not explicitly specified)
param_markers.minDistanceToBorder = 3
param_markers.minOtsuStdDev = 1.0  # Default value (not explicitly specified)

# Perform marker detection
marker_corners, marker_IDs, reject = aruco.detectMarkers(
    gray_frame, marker_dict, parameters=param_markers
)

if marker_corners:
    min_distance = np.inf
    closest_marker_ID = None

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

        bottom_left_frame = np.array([0, frame.shape[0]])
        distance_to_bottom_left = np.linalg.norm(
            bottom_left_frame - bottom_left)

        if i == 0 or distance_to_bottom_left < min_distance:
            min_distance = distance_to_bottom_left
            closest_marker_ID = ids[0]

        # Display ID and XYZ pose of each detected marker
        marker_id = ids[0]
        marker_pose = f"Marker ID: {marker_id}, XYZ Pose: {tVec[i][0]}"
        # cv.putText(frame, marker_pose, (corners[0][0], corners[0]
        #            [1] - 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

    if closest_marker_ID:
        closest_marker_index = np.where(marker_IDs == closest_marker_ID)[0][0]
        closest_marker_corners = marker_corners[closest_marker_index]
        cv.polylines(
            frame,
            [closest_marker_corners.astype(np.int32)],
            True,
            (0, 255, 0),
            4,
            cv.LINE_AA,
        )

# Display the resulting image with markers highlighted and XYZ poses
cv.imshow("Detected Markers with XYZ Poses", frame)
cv.waitKey(0)
cv.destroyAllWindows()
