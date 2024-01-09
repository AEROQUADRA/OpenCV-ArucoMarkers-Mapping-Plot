import cv2
import numpy as np

# Load the image
image = cv2.imread('distance_check/Store/untitled6.png')

# Define the coordinates of the document's corners in the input image
# Order of points: top-left, top-right, bottom-right, bottom-left
input_points = np.array(
    [[565, 203], [1197, 245], [1429, 834], [590, 970]], dtype=np.float32)

# Define the size of the output image (width, height) - adjust as needed
width, height = 600, 800
output_points = np.array(
    [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype=np.float32)

# Compute the perspective transformation matrix
matrix = cv2.getPerspectiveTransform(input_points, output_points)

# Apply the perspective transformation to the input image
output_image = cv2.warpPerspective(image, matrix, (width, height))

# Display the original and transformed images
cv2.imshow('Original Image', image)
cv2.imshow('Perspective Transformation', output_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
