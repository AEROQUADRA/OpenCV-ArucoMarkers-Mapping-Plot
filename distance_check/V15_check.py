import cv2
import numpy as np

# Function to determine the grid position of the detected ArUco marker


def determine_grid_position(corners, image_size, grid_size):
    # Compute the mean of the corners to get the center of the ArUco marker
    center = np.mean(corners, axis=1)[0]
    # Ensure center is an array with two elements
    if center.shape != (2,):
        raise ValueError(f"Center should be a 2-element array, got {center}")
    # Determine the size of each grid cell
    cell_size_x = image_size[1] / grid_size[1]
    cell_size_y = image_size[0] / grid_size[0]
    # Compute the position in the grid
    grid_x = int(center[0] // cell_size_x)
    grid_y = int(center[1] // cell_size_y)
    return grid_y, grid_x  # Return as row, column


# Load the image
image_path = 'distance_check/Store/untitled5.png'  # Replace with your image path
image = cv2.imread(image_path)

# Check if the image is loaded properly
if image is None:
    raise ValueError("Image not found. Check the path.")

# Convert image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect ArUco markers
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
parameters = cv2.aruco.DetectorParameters()
corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(
    gray, aruco_dict, parameters=parameters)

# Grid size needs to be determined dynamically, for now we assume a 5x5 grid as an example
# We will also need to determine the actual grid size by analyzing the image
grid_size = (5, 5)  # Placeholder for actual grid size

# Initialize a grid to store ArUco IDs
aruco_grid = [[0 for _ in range(grid_size[1])] for _ in range(grid_size[0])]

# Assuming markers are detected, place them in the grid
if ids is not None:
    for corner, id in zip(corners, ids.flatten()):
        # Get the grid position
        grid_position = determine_grid_position(corner, image.shape, grid_size)
        # Place the id in the grid
        row, col = grid_position
        aruco_grid[row][col] = id

# Convert the grid to the desired output format
output = "\n".join("".join(str(cell).ljust(5) for cell in row)
                   for row in aruco_grid)
print(output)
