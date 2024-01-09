import cv2
import numpy as np

# Function to find 4x4 250 ArUco markers and their corners


def find_aruco_markers(image_path):
    # Read the image
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Define the dictionary of 4x4 250 ArUco markers
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
    aruco_params = cv2.aruco.DetectorParameters()

    # Detect the markers in the image
    corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(
        gray, aruco_dict, parameters=aruco_params)

    if ids is not None and len(ids) >= 4:
        # Sort detected markers by their ids
        sorted_indices = np.argsort(ids, axis=0).flatten()
        sorted_corners = [corners[i][0] for i in sorted_indices]

        # Find corners closest to the image corners
        image_corners = np.array([[0, 0], [gray.shape[1], 0], [
                                 gray.shape[1], gray.shape[0]], [0, gray.shape[0]]], dtype=np.float32)
        closest_markers = []
        for img_corner in image_corners:
            distances = [np.linalg.norm(
                np.mean(c, axis=0) - img_corner) for c in sorted_corners]
            closest_marker = sorted_corners[np.argmin(distances)]
            closest_markers.append(closest_marker)

        return closest_markers
    else:
        return None


# Apply the function to find corners in your image
image_path = 'distance_check/Store/p0.jpg'
corners_coordinates = find_aruco_markers(image_path)

# Check if we found valid corners
if corners_coordinates and len(corners_coordinates) == 4:
    # Load the image
    image = cv2.imread(image_path)

    # Define the coordinates of the document's corners in the input image
    input_points = np.array([c[0]
                            for c in corners_coordinates], dtype=np.float32)

    # Define the size of the output image along with margin (width, height)
    margin = 500  # You can adjust this margin value as needed
    width, height = 600 + 2 * margin, 800 + 2 * \
        margin  # Add margin to width and height

    # Define the output points with margin
    output_points = np.array([
        [margin, margin],
        [width - 1 - margin, margin],
        [width - 1 - margin, height - 1 - margin],
        [margin, height - 1 - margin]
    ], dtype=np.float32)

    # Compute the perspective transformation matrix
    matrix = cv2.getPerspectiveTransform(input_points, output_points)

    # Apply the perspective transformation to the input image
    output_image = cv2.warpPerspective(image, matrix, (width, height))

    # Display the original and transformed images
    cv2.imshow('Original Image', image)
    cv2.imshow('Perspective Transformation', output_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("Could not find enough markers or an error occurred.")
