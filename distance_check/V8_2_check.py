import cv2 as cv
from cv2 import aruco
import numpy as np
from picamera2 import Picamera2
import RPi.GPIO as GPIO
from gpiozero import PWMOutputDevice, DigitalOutputDevice
import time
import curses

# Motor A, Left Side GPIO CONSTANTS
PWM_DRIVE_LEFT = 12       # ENA - H-Bridge enable pin
FORWARD_LEFT_PIN = 20      # IN1 - Forward Drive
REVERSE_LEFT_PIN = 16      # IN2 - Reverse Drive

# Motor B, Right Side GPIO CONSTANTS
PWM_DRIVE_RIGHT = 13       # ENB - H-Bridge enable pin
FORWARD_RIGHT_PIN = 26     # IN1 - Forward Drive
REVERSE_RIGHT_PIN = 21      # IN2 - Reverse Drive

# Initialize motor objects
driveLeft = PWMOutputDevice(PWM_DRIVE_LEFT)
driveRight = PWMOutputDevice(PWM_DRIVE_RIGHT)
forwardLeft = DigitalOutputDevice(FORWARD_LEFT_PIN)
reverseLeft = DigitalOutputDevice(REVERSE_LEFT_PIN)
forwardRight = DigitalOutputDevice(FORWARD_RIGHT_PIN)
reverseRight = DigitalOutputDevice(REVERSE_RIGHT_PIN)

# Initialize time variables
prev_time = time.time()

# Initialize GPIO
GPIO.setmode(GPIO.BCM)

# Cleanup GPIO on program exit
def cleanup_gpio():
    GPIO.cleanup()

def allStop():
    forwardLeft.off()
    reverseLeft.off()
    forwardRight.off()
    reverseRight.off()
    driveLeft.value = 0
    driveRight.value = 0

def forwardDrive(speed_left, speed_right):
    forwardLeft.on()
    reverseLeft.off()
    forwardRight.on()
    reverseRight.off()
    driveLeft.value = speed_left
    driveRight.value = speed_right

def reverseDrive(speed_left, speed_right):
    forwardLeft.off()
    reverseLeft.on()
    forwardRight.off()
    reverseRight.on()
    driveLeft.value = speed_left
    driveRight.value = speed_right

def spinLeft(speed_left, speed_right):
    forwardLeft.off()
    reverseLeft.on()
    forwardRight.on()
    reverseRight.off()
    driveLeft.value = speed_left
    driveRight.value = speed_right

def spinRight(speed_left, speed_right):
    forwardLeft.on()
    reverseLeft.off()
    forwardRight.off()
    reverseRight.on()
    driveLeft.value = speed_left
    driveRight.value = speed_right

def arucoDetect():
    picam2 = Picamera2()
    picam2.preview_configuration.main.size = (800, 800)
    picam2.preview_configuration.main.format = "RGB888"
    picam2.preview_configuration.align()
    picam2.configure("preview")
    picam2.start()

    # Load calibration data
    calib_data_path = "calib_data/calibration_data.npz"
    calib_data = np.load(calib_data_path)
    cam_mat = calib_data["cameraMatrix"]
    dist_coef = calib_data["distCoeffs"]
    r_vectors = calib_data["rvecs"]
    t_vectors = calib_data["tvecs"]

    MARKER_SIZE = 5  # centimeters (measure your printed marker size)
    marker_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_250)
    param_markers = aruco.DetectorParameters()

    try:
        while True:
            frame = picam2.capture_array()

            # Rotate the frame 180 degrees
            frame = cv.rotate(frame, cv.ROTATE_180)

            # Convert the frame to grayscale
            gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

            height, width = gray_frame.shape[:2]
            cv.line(gray_frame, (int(width / 2) - 10, int(height / 2)),
                    (int(width / 2) + 10, int(height / 2)), (255, 0, 0), 2)
            cv.line(gray_frame, (int(width / 2), int(height / 2) - 10),
                    (int(width / 2), int(height / 2) + 10), (255, 0, 0), 2)
            cv.putText(gray_frame, "0,0", (int(width / 2) + 5, int(height / 2) - 5),
                       cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            marker_corners, marker_IDs, _ = aruco.detectMarkers(
                gray_frame, marker_dict, parameters=param_markers
            )

            if marker_corners:
                rVec, tVec, _ = aruco.estimatePoseSingleMarkers(
                    marker_corners, MARKER_SIZE, cam_mat, dist_coef
                )

                # Assuming only one marker is detected
                ids, corners, i = marker_IDs[0], marker_corners[0], 0

                cv.polylines(
                    gray_frame, [corners.astype(np.int32)], True, (0,
                                                                   255, 255), 4, cv.LINE_AA
                )
                corners = corners.reshape(4, 2)
                corners = corners.astype(int)
                top_right = corners[0].ravel()
                bottom_right = corners[2].ravel()

                distance = np.sqrt(
                    tVec[i][0][2] ** 2 +
                    tVec[i][0][0] ** 2 + tVec[i][0][1] ** 2
                )

                cv.drawFrameAxes(
                    gray_frame, cam_mat, dist_coef, rVec[i], tVec[i], MARKER_SIZE * 0.6
                )

                cv.putText(gray_frame, f"id: {ids[0]} Dist: {round(distance, 2)}",
                           top_right,
                           cv.FONT_HERSHEY_PLAIN,
                           1.3,
                           (0, 0, 255),
                           2,
                           cv.LINE_AA,
                           )
                cv.putText(gray_frame, f"x:{round(tVec[i][0][0], 1)} y: {round(tVec[i][0][1], 1)} ",
                           bottom_right,
                           cv.FONT_HERSHEY_PLAIN,
                           1.0,
                           (0, 0, 255),
                           2,
                           cv.LINE_AA,
                           )

                rMat, _ = cv.Rodrigues(rVec[i])
                angles = cv.RQDecomp3x3(rMat)[0]
                cv.putText(gray_frame, f"Roll: {round(angles[0], 2)} Pitch: {round(angles[1], 2)} Yaw: {round(angles[2], 2)}",
                           (bottom_right[0], bottom_right[1] + 30),
                           cv.FONT_HERSHEY_PLAIN,
                           1.0,
                           (0, 0, 255),
                           2,
                           cv.LINE_AA,
                           )

                detectedId = ids[0]

                # Output the distance to the detected marker
                return distance, detectedId

            cv.imshow("gray_frame", gray_frame)

    finally:
        # Ensure the camera is closed before returning
        picam2.close()
        cv.destroyAllWindows()

def move(win, distance, detectedId):
    distanceRounded = round(distance, 2)
    start_time = time.time()
    while time.time() - start_time < distanceRounded:
        if distanceRounded > 70:
            print(
                f"Rejected - Distance to marker {detectedId}: {distanceRounded} (greater than 70 cm)")
            distance, detectedId = arucoDetect()
            start_time = time.time()  # Reset the start time for the new distance
            break

    forwardDrive(forward_speed_left, forward_speed_right)
    print(f"Approaching ID {detectedId} in {distanceRounded} cm \n")
    win.refresh()

    # Main speed
    speedMain = 0.3
    # Adjust these values for the desired speeds
    forward_speed_left = speedMain
    forward_speed_right = speedMain
    reverse_speed_left = speedMain
    reverse_speed_right = speedMain
    turn_left_speed_left = speedMain
    turn_left_speed_right = speedMain
    turn_right_speed_left = speedMain
    turn_right_speed_right = speedMain

    print(f"Distance to marker {detectedId}: {distanceRounded}")

    start_time = time.time()
    while time.time() - start_time < distanceRounded/10:
        forwardDrive(forward_speed_left, forward_speed_right)
        print(f"Approaching ID {detectedId} in {distanceRounded} cm \n")
        win.refresh()

    allStop()

    while True:
        current_time = time.time()
        time_elapsed = current_time - prev_time

        key = win.getch()
        if key == curses.KEY_UP:
            forwardDrive(forward_speed_left, forward_speed_right)
            win.addstr("Forward")
        elif key == curses.KEY_DOWN:
            reverseDrive(reverse_speed_left, reverse_speed_right)
            win.addstr("Reverse")
        elif key == curses.KEY_LEFT:
            spinLeft(turn_left_speed_left, turn_left_speed_right)
            win.addstr("Spin Left")
        elif key == curses.KEY_RIGHT:
            spinRight(turn_right_speed_left, turn_right_speed_right)
            win.addstr("Spin Right")
        else:
            allStop()
            win.addstr("Stop")

        win.refresh()
        prev_time = current_time

def main(win):

    # Adjust these values for the desired speeds
    forward_speed_left = 0.3
    forward_speed_right = 0.3
    reverse_speed_left = 0.3
    reverse_speed_right = 0.3
    turn_left_speed_left = 0.3
    turn_left_speed_right = 0.3
    turn_right_speed_left = 0.3
    turn_right_speed_right = 0.3

    # Call the function and unpack the returned values
    result = arucoDetect()

    if result is not None:
        distance, detectedId = result
        move(win, distance, detectedId)
    else:
        print("No markers detected. Error in arucoDetect function")

# Run the main function with GPIO cleanup on exit
if __name__ == "__main__":
    try:
        curses.wrapper(main)
    finally:
        cleanup_gpio()
