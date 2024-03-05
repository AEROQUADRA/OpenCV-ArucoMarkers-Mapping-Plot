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

MARKER_SIZE = 5  # centimeters (measure your printed marker size)

marker_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_250)

# Load your image
image_path = 'distance_check/Store/tr3.jpg'  # Replace with your image path
frame = cv.imread(image_path)

if frame is not None:
    # Convert the image to grayscale
    gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # Adjust parameters to increase sensitivity
    param_markers = aruco.DetectorParameters()

    param_markers.adaptiveThreshConstant = 1  # Try lowering this value
    param_markers.minMarkerPerimeterRate = 0.001  # Try lowering this value
    param_markers.maxMarkerPerimeterRate = 1.0  # Try increasing this value
    param_markers.polygonalApproxAccuracyRate = 0.1  # Try lowering this value
    param_markers.minCornerDistanceRate = 0.05  # Try lowering this value
    param_markers.minDistanceToBorder = 1  # Try lowering this value
    param_markers.minOtsuStdDev = 1.0  # Try lowering this value

    # Perform marker detection
    marker_corners, marker_IDs, reject = aruco.detectMarkers(
        gray_frame, marker_dict, parameters=param_markers
    )

    if marker_corners:
        rVec, tVec, _ = aruco.estimatePoseSingleMarkers(
            marker_corners, MARKER_SIZE, cam_mat, dist_coef
        )

        total_markers = range(0, marker_IDs.size)
        for ids, corners, i in zip(marker_IDs, marker_corners, total_markers):
            cv.polylines(
                gray_frame, [corners.astype(np.int32)], True, (0,
                                                               255, 255), 4, cv.LINE_AA
            )

            corners = corners.reshape(4, 2)
            corners = corners.astype(int)
            top_right = corners[0].ravel()
            top_left = corners[1].ravel()
            bottom_right = corners[2].ravel()
            bottom_left = corners[3].ravel()

            # Calculating the distance
            distance = np.sqrt(
                tVec[i][0][2] ** 2 + tVec[i][0][0] ** 2 + tVec[i][0][1] ** 2
            )
            # Draw the pose of the marker
            point = cv.drawFrameAxes(
                gray_frame, cam_mat, dist_coef, rVec[i], tVec[i], 4, 4)
            cv.putText(
                gray_frame,
                f"id: {ids[0]} Dist: {round(distance, 2)}",
                top_right,
                cv.FONT_HERSHEY_PLAIN,
                1.3,
                (0, 0, 255),
                2,
                cv.LINE_AA,
            )
            cv.putText(
                gray_frame,
                f"x:{round(tVec[i][0][0],1)} y: {round(tVec[i][0][1],1)} ",
                bottom_right,
                cv.FONT_HERSHEY_PLAIN,
                1.0,
                (0, 0, 255),
                2,
                cv.LINE_AA,
            )
        cv.imshow("gframe", gray_frame)
        cv.waitKey(0)  # Wait indefinitely until any key is pressed
        cv.destroyAllWindows()
    else:
        print("No markers found in the image.")
else:
    print("Image not found or could not be read.")
