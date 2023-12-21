import matplotlib.pyplot as plt
import numpy as np
import cv2  # Ensure OpenCV is installed: pip install opencv-python

# Load the calibration data from the .npz file
calibration_data = np.load("calib_data/calibration_data.npz")

# Access the arrays inside the loaded file
camera_matrix = calibration_data['camera_matrix']
dist_coeffs = calibration_data['dist_coeffs']

# Marker data (replace with your marker data)
marker_data = [
    {"id": 7, "x": -0.3, "y": 0.2, "distance": 30.04,
        "roll": 160.34, "pitch": -26.07, "yaw": -70.02},
    {"id": 4, "x": -6.9, "y": -0.9, "distance": 32.34,
        "roll": 156.66, "pitch": -26.52, "yaw": -69.66},
    {"id": 1, "x": -13.0, "y": -2.0, "distance": 36.28,
        "roll": 159.44, "pitch": -26.38, "yaw": -69.8},
    {"id": 5, "x": -4.8, "y": -6.5, "distance": 35.54,
        "roll": 157.28, "pitch": -26.01, "yaw": -70.21},
    {"id": 2, "x": -11.0, "y": -7.7, "distance": 39.4,
        "roll": 159.82, "pitch": -24.48, "yaw": -70.45},
    {"id": 6, "x": -2.9, "y": -12.3, "distance": 39.59,
        "roll": 157.61, "pitch": -25.12, "yaw": -71.11},
    {"id": 3, "x": -9.0, "y": -13.4, "distance": 42.89,
        "roll": 161.2, "pitch": -25.29, "yaw": -69.93},
    {"id": 8, "x": 1.7, "y": -5.4, "distance": 32.48,
        "roll": 159.05, "pitch": -27.24, "yaw": -70.15},
    {"id": 0, "x": 3.7, "y": -11.1, "distance": 37.15,
        "roll": 160.55, "pitch": -25.37, "yaw": -70.15}
]

# Define grid layout (3x3)
grid_layout = [
    (0, 0), (1, 0), (2, 0),
    (0, 1), (1, 1), (2, 1),
    (0, 2), (1, 2), (2, 2)
]

# Correct marker positions based on grid layout and camera matrix
corrected_positions = []

for marker in marker_data:
    marker_id = marker["id"]
    marker_x = marker["x"]
    marker_y = marker["y"]
    grid_x, grid_y = grid_layout[marker_id]

    # Apply distortion correction
    distorted_points = np.array([[marker_x, marker_y]], dtype='float32')
    undistorted_points = cv2.undistortPoints(
        distorted_points, camera_matrix, dist_coeffs)

    corrected_x = undistorted_points[0][0][0]
    corrected_y = undistorted_points[0][0][1]

    corrected_positions.append((corrected_x + grid_x, corrected_y + grid_y))

# Plotting corrected marker positions
plt.figure(figsize=(8, 6))
for i, position in enumerate(corrected_positions):
    plt.scatter(position[0], position[1], label=f"Marker {i}")

plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Corrected Marker Positions in Grid')
plt.legend()
plt.grid(True)
plt.show()
