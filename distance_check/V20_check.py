import cv2
import numpy as np
import matplotlib.pyplot as plt
from cv2 import aruco

# Function to find ArUco-like markers


def find_aruco_like_markers(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(
        gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(
        binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    def compute_centroid(corners):
        M = cv2.moments(corners)
        if M['m00'] == 0:
            return (0, 0)
        return (int(M['m10'] / M['m00']), int(M['m01'] / M['m00']))

    def sort_corners(corners):
        rect = np.zeros((4, 2), dtype="float32")
        s = corners.sum(axis=1)
        rect[0] = corners[np.argmin(s)]
        rect[2] = corners[np.argmax(s)]
        diff = np.diff(corners, axis=1)
        rect[1] = corners[np.argmin(diff)]
        rect[3] = corners[np.argmax(diff)]
        return rect

    def identify_corner_type(centroid, image_shape):
        cx, cy = centroid
        w, h = image_shape[1], image_shape[0]
        if cx <= w / 2 and cy <= h / 2:
            return "Top Left"
        elif cx > w / 2 and cy <= h / 2:
            return "Top Right"
        elif cx <= w / 2 and cy > h / 2:
            return "Bottom Left"
        elif cx > w / 2 and cy > h / 2:
            return "Bottom Right"
        return "Unknown"

    aruco_like_contours = []
    for c in contours:
        perimeter = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.04 * perimeter, True)
        if len(approx) == 4:
            aruco_like_contours.append(approx)

    aruco_like_contours = [
        c for c in aruco_like_contours if cv2.contourArea(c) > 100]

    if len(aruco_like_contours) >= 4:
        image_corners = [
            np.array([0, 0]),
            np.array([gray.shape[1], 0]),
            np.array([gray.shape[1], gray.shape[0]]),
            np.array([0, gray.shape[0]])
        ]

        closest_contours = []
        for img_corner in image_corners:
            distances = [np.linalg.norm(compute_centroid(
                c.reshape(4, 2)) - img_corner) for c in aruco_like_contours]
            closest_contour = aruco_like_contours[np.argmin(distances)]
            sorted_corner_pts = sort_corners(closest_contour.reshape(4, 2))
            closest_contours.append({
                'centroid': compute_centroid(closest_contour.reshape(4, 2)),
                'corners': sorted_corner_pts.tolist()
            })

        coordinates = []
        for contour in closest_contours:
            centroid = contour['centroid']
            corners = contour['corners']
            corner_type = identify_corner_type(centroid, gray.shape)

            if corner_type == "Top Left":
                relevant_corner = corners[0]
            elif corner_type == "Top Right":
                relevant_corner = corners[1]
            elif corner_type == "Bottom Right":
                relevant_corner = corners[2]
            elif corner_type == "Bottom Left":
                relevant_corner = corners[3]

            coordinates.append(relevant_corner)

        return coordinates
    else:
        return None


# Function to detect ArUco markers using OpenCV's built-in functions
def detect_aruco_markers(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_250)
    aruco_params = aruco.DetectorParameters()
    corners, ids, _ = aruco.detectMarkers(
        gray, aruco_dict, parameters=aruco_params)
    return corners, ids

# Function to plot markers on a chart


def plot_markers(corners, ids, image_shape, ax):
    for corner, id in zip(corners, ids.flatten()):
        centroid = np.mean(corner[0], axis=0)
        # Inverting y for correct plot orientation
        ax.scatter(centroid[0], image_shape[0] - centroid[1])
        ax.text(centroid[0], image_shape[0] -
                centroid[1], f'ID: {id}', color='red')

    ax.set_xlim([0, image_shape[1]])
    ax.set_ylim([0, image_shape[0]])
    ax.invert_yaxis()  # Inverting y-axis for image coordinates
    ax.set_title('ArUco Markers Position')
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')

# Function to draw markers on an image


def draw_markers_on_image(image, corners, ids):
    return aruco.drawDetectedMarkers(image.copy(), corners, ids)

# Main function


def main():
    # Change this to your image path
    image_path = 'distance_check/Store/untitled6.png'
    original_image = cv2.imread(image_path)

    # Find ArUco-like markers and apply perspective transformation
    corners_coordinates = find_aruco_like_markers(original_image)
    if corners_coordinates and len(corners_coordinates) == 4:
        input_points = np.array(corners_coordinates, dtype=np.float32)
        margin = 500
        width, height = 600 + 2 * margin, 800 + 2 * margin
        output_points = np.array([
            [margin, margin],
            [width - 1 - margin, margin],
            [width - 1 - margin, height - 1 - margin],
            [margin, height - 1 - margin]
        ], dtype=np.float32)

        matrix = cv2.getPerspectiveTransform(input_points, output_points)
        corrected_image = cv2.warpPerspective(
            original_image, matrix, (width, height))
    else:
        print("Could not find enough markers or an error occurred.")
        return

    # Detect ArUco markers in the corrected image
    corners, ids = detect_aruco_markers(corrected_image)

    if ids is not None:
        # Draw markers on the corrected image
        marked_image = draw_markers_on_image(corrected_image, corners, ids)

        # Plotting all images and the chart
        fig, axs = plt.subplots(2, 2, figsize=(10, 8))

        axs[0, 0].imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
        axs[0, 0].set_title('Original Image')
        axs[0, 0].axis('off')

        axs[0, 1].imshow(cv2.cvtColor(corrected_image, cv2.COLOR_BGR2RGB))
        axs[0, 1].set_title('Corrected Image')
        axs[0, 1].axis('off')

        axs[1, 0].imshow(cv2.cvtColor(marked_image, cv2.COLOR_BGR2RGB))
        axs[1, 0].set_title('Marked Image')
        axs[1, 0].axis('off')

        plot_markers(corners, ids, corrected_image.shape, axs[1, 1])

        plt.tight_layout()
        plt.show()
    else:
        print("No ArUco markers detected.")


if __name__ == "__main__":
    main()