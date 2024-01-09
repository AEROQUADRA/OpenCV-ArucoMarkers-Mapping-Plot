import cv2
import numpy as np


def find_aruco_like_markers(image_path):
    # Read the image
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Thresholding to get a binary image
    _, binary = cv2.threshold(
        gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # Find contours
    contours, _ = cv2.findContours(
        binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Function to compute the centroid of corners
    def compute_centroid(corners):
        M = cv2.moments(corners)
        if M['m00'] == 0:
            return (0, 0)
        return (int(M['m10']/M['m00']), int(M['m01']/M['m00']))

    # Sort the corners to get top-left, top-right, bottom-right, bottom-left
    def sort_corners(corners):
        rect = np.zeros((4, 2), dtype="float32")
        s = corners.sum(axis=1)
        rect[0] = corners[np.argmin(s)]
        rect[2] = corners[np.argmax(s)]
        diff = np.diff(corners, axis=1)
        rect[1] = corners[np.argmin(diff)]
        rect[3] = corners[np.argmax(diff)]
        return rect

    # Function to identify the corner type
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

    # Approximate the contours to polygons and get only the rectangular ones
    aruco_like_contours = []
    for c in contours:
        perimeter = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.04 * perimeter, True)
        if len(approx) == 4:
            aruco_like_contours.append(approx)

    # Filter out very small contours that are unlikely to be Aruco markers
    aruco_like_contours = [
        c for c in aruco_like_contours if cv2.contourArea(c) > 100]

    # If we have at least 4 contours, find the ones closest to each corner of the image
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

            print(f"{corner_type} Marker:")
            print(f"Relevant Corner (x, y): {relevant_corner}\n")

    else:
        return None


# Use the function with your image path
# Replace with your actual image path
image_path = 'distance_check/Store/untitled6.png'
result = find_aruco_like_markers(image_path)
