import cv2
import numpy as np
import matplotlib.pyplot as plt


def find_aruco_like_markers(image_path):
    image = cv2.imread(image_path)
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


image_path = 'distance_check/Store/untitled5.png'
corners_coordinates = find_aruco_like_markers(image_path)

if corners_coordinates and len(corners_coordinates) == 4:
    image = cv2.imread(image_path)
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
    output_image = cv2.warpPerspective(image, matrix, (width, height))

    detected_markers_image = np.copy(image)
    for corner in corners_coordinates:
        x, y = map(int, corner)
        cv2.circle(detected_markers_image, (x, y), 8, (0, 255, 0), -1)

    plt.figure(figsize=(8, 6))
    for corner in corners_coordinates:
        plt.scatter(corner[0], corner[1], color='red')
    plt.title('Detected ArUco Markers')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.gca().invert_yaxis()
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    fig, axs = plt.subplots(2, 2, figsize=(12, 10))

    axs[0, 0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axs[0, 0].set_title('Original Image')

    axs[0, 1].imshow(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
    axs[0, 1].set_title('Perspective Transformation')

    axs[1, 0].imshow(cv2.cvtColor(detected_markers_image, cv2.COLOR_BGR2RGB))
    axs[1, 0].set_title('Detected Markers')

    axs[1, 1].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    for corner in corners_coordinates:
        x, y = map(int, corner)
        axs[1, 1].scatter(x, y, color='red')
    axs[1, 1].set_title('Detected ArUco Markers')

    plt.tight_layout()
    plt.show()
else:
    print("Could not find enough markers or an error occurred.")
