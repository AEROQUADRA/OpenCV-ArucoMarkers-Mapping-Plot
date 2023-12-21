import cv2 as cv
import os
import numpy as np

# Chess/checker board size, dimensions
CHESS_BOARD_DIM = (9, 6)

# The size of squares in the checker board design.
SQUARE_SIZE = 38  # millimeters (change it according to printed size)

# Termination criteria for corner detection
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Directory path to save calibration data
calib_data_path = "calib_data"

# Check if the directory exists, if not, create it
if not os.path.isdir(calib_data_path):
    os.makedirs(calib_data_path)
    print(f'"{calib_data_path}" Directory is created')
else:
    print(f'"{calib_data_path}" Directory already exists.')

# Prepare object points in 3D space
obj_3D = np.zeros((CHESS_BOARD_DIM[0] * CHESS_BOARD_DIM[1], 3), np.float32)
obj_3D[:, :2] = np.mgrid[0:CHESS_BOARD_DIM[0],
                         0:CHESS_BOARD_DIM[1]].T.reshape(-1, 2)
obj_3D *= SQUARE_SIZE
print(obj_3D)

# Arrays to store object points and image points from all the given images.
obj_points_3D = []  # 3D point in real-world space
img_points_2D = []  # 2D points in image plane

# Update the image directory path to reflect its correct location inside 'camera_calibration'
image_dir_path = "camera_calibration/images"

# List all files in the images directory
files = os.listdir(image_dir_path)

# Process each image in the directory
for file in files:
    print(file)
    imagePath = os.path.join(image_dir_path, file)

    image = cv.imread(imagePath)
    grayScale = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    ret, corners = cv.findChessboardCorners(image, CHESS_BOARD_DIM, None)

    if ret:
        obj_points_3D.append(obj_3D)
        corners2 = cv.cornerSubPix(
            grayScale, corners, (3, 3), (-1, -1), criteria)
        img_points_2D.append(corners2)

        img = cv.drawChessboardCorners(image, CHESS_BOARD_DIM, corners2, ret)

cv.destroyAllWindows()

# Camera calibration
h, w = grayScale.shape[:2]
ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(
    obj_points_3D, img_points_2D, (w, h), None, None
)

print("Calibration completed")

# Save calibration data using numpy
np.savez(
    f"{calib_data_path}/calibration_data.npz",
    cameraMatrix=mtx,
    distCoeffs=dist,
    rvecs=rvecs,
    tvecs=tvecs,
)

print("Calibration data saved successfully")

# Load calibration data
loaded_data = np.load(f"{calib_data_path}/calibration_data.npz")

loaded_camera_matrix = loaded_data["cameraMatrix"]
loaded_dist_coeffs = loaded_data["distCoeffs"]
loaded_rvecs = loaded_data["rvecs"]
loaded_tvecs = loaded_data["tvecs"]

print("Calibration data loaded successfully")
