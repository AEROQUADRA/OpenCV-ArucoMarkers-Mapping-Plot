import cv2 as cv
from cv2 import aruco
import numpy as np
from picamera2 import Picamera2
import RPi.GPIO as GPIO
from gpiozero import PWMOutputDevice, DigitalOutputDevice
from time import time, sleep
import math
import py_qmc5883l
import curses


# Motor A, Left Side GPIO CONSTANTS
PWM_DRIVE_LEFT = 12  # ENA - H-Bridge enable pin
FORWARD_LEFT_PIN = 26  # IN1 - Forward Drive
REVERSE_LEFT_PIN = 19  # IN2 - Reverse Drive

# Motor B, Right Side GPIO CONSTANTS
PWM_DRIVE_RIGHT = 13  # ENB - H-Bridge enable pin
FORWARD_RIGHT_PIN = 21  # IN1 - Forward Drive
REVERSE_RIGHT_PIN = 20  # IN2 - Reverse Drive

# GPIO pin for the buzzer
BUZZER_PIN = 16


# Initialize motor objects
driveLeft = PWMOutputDevice(PWM_DRIVE_LEFT)
driveRight = PWMOutputDevice(PWM_DRIVE_RIGHT)
forwardLeft = DigitalOutputDevice(FORWARD_LEFT_PIN)
reverseLeft = DigitalOutputDevice(REVERSE_LEFT_PIN)
forwardRight = DigitalOutputDevice(FORWARD_RIGHT_PIN)
reverseRight = DigitalOutputDevice(REVERSE_RIGHT_PIN)

sensor = py_qmc5883l.QMC5883L()
sensor.calibration = [[1.0256545432028572, -0.013013085106172594, 289.7145364429366],
                      [-0.01301308510617258, 1.0066007951356402, 879.0513837979042], [0.0, 0.0, 1.0]]

# Initialize time variables
prev_time = time()

# Initialize GPIO
GPIO.setmode(GPIO.BCM)
GPIO.setup(BUZZER_PIN, GPIO.OUT)


def beep(beep_time):
    GPIO.output(BUZZER_PIN, GPIO.HIGH)
    sleep(beep_time)  # Keep GPIO BUZZER_PIN high for beep_time seconds
    GPIO.output(16, GPIO.LOW)


def double_beep(beep_time):
    GPIO.output(BUZZER_PIN, GPIO.HIGH)
    sleep(beep_time)  # Keep GPIO BUZZER_PIN high for beep_time seconds
    GPIO.output(16, GPIO.LOW)
    sleep(0.1)
    GPIO.output(BUZZER_PIN, GPIO.HIGH)
    sleep(beep_time)
    GPIO.output(16, GPIO.LOW)


def tri_beep(beep_time):
    GPIO.output(BUZZER_PIN, GPIO.HIGH)
    sleep(beep_time)  # Keep GPIO BUZZER_PIN high for beep_time seconds
    GPIO.output(16, GPIO.LOW)
    sleep(0.1)
    GPIO.output(BUZZER_PIN, GPIO.HIGH)
    sleep(beep_time)
    GPIO.output(16, GPIO.LOW)
    sleep(0.1)
    GPIO.output(BUZZER_PIN, GPIO.HIGH)
    sleep(beep_time)
    GPIO.output(16, GPIO.LOW)


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


def forceBrake():
    # Quick forward pulse
    forwardLeft.on()
    reverseLeft.off()
    forwardRight.on()
    reverseRight.off()
    driveLeft.value = 0.5
    driveRight.value = 0.5
    sleep(0.03)

    # Quick backward pulse
    forwardLeft.off()
    reverseLeft.on()
    forwardRight.off()
    reverseRight.on()
    driveLeft.value = 0.5
    driveRight.value = 0.5
    sleep(0.03)

    # Quick forward pulse
    forwardLeft.on()
    reverseLeft.off()
    forwardRight.on()
    reverseRight.off()
    driveLeft.value = 0.5
    driveRight.value = 0.5
    sleep(0.03)

    # Quick backward pulse
    forwardLeft.off()
    reverseLeft.on()
    forwardRight.off()
    reverseRight.on()
    driveLeft.value = 0.5
    driveRight.value = 0.5
    sleep(0.03)

    # Stop the motors
    allStop()


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


def get_heading():
    bearing = sensor.get_bearing()
    full_value_bearing = int(bearing)
    return full_value_bearing


def spin_to_heading(speed, desired_heading, threshold):
    while True:
        current_heading = get_heading()
        if current_heading is not None:
            if (current_heading < (desired_heading - threshold) % 360 or
                    current_heading > (desired_heading + threshold) % 360):
                # Determine the direction to spin
                if (current_heading - desired_heading + 180) % 360 < 180:
                    # Spin left
                    spinLeft(speed, speed)
                else:
                    # Spin right
                    spinRight(speed, speed)
            else:
                break
        else:
            print("Error: Unable to get current heading")
            break


def point_direction(rotation_speed, desired_heading, threshold):
    spin_to_heading(rotation_speed, desired_heading, threshold)
    forceBrake()


def point_north(rotation_speed, threshold, start_heading):
    desired_heading = (start_heading - 90) % 360
    point_direction(rotation_speed, desired_heading, threshold)


def point_east(rotation_speed, threshold, start_heading):
    desired_heading = (start_heading - 0) % 360
    point_direction(rotation_speed, desired_heading, threshold)


def point_south(rotation_speed, threshold, start_heading):
    desired_heading = (start_heading + 90) % 360
    point_direction(rotation_speed, desired_heading, threshold)


def point_west(rotation_speed, threshold, start_heading):
    desired_heading = (start_heading - 180) % 360
    point_direction(rotation_speed, desired_heading, threshold)


def point_north_east(rotation_speed, threshold, start_heading):
    desired_heading = (start_heading - 45) % 360
    point_direction(rotation_speed, desired_heading, threshold)


def point_south_east(rotation_speed, threshold, start_heading):
    desired_heading = (start_heading + 45) % 360
    point_direction(rotation_speed, desired_heading, threshold)


def point_north_west(rotation_speed, threshold, start_heading):
    desired_heading = (start_heading - 135) % 360
    point_direction(rotation_speed, desired_heading, threshold)


def point_south_west(rotation_speed, threshold, start_heading):
    desired_heading = (start_heading + 135) % 360
    point_direction(rotation_speed, desired_heading, threshold)


def detect_aruco_marker(picam2, calib_data_path="calib_data/calibration_data.npz"):
    # Load calibration data
    calib_data = np.load(calib_data_path)
    cam_mat = calib_data["cameraMatrix"]
    dist_coef = calib_data["distCoeffs"]
    r_vectors = calib_data["rvecs"]
    t_vectors = calib_data["tvecs"]

    MARKER_SIZE = 5  # centimeters (measure your printed marker size)
    marker_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_250)
    param_markers = aruco.DetectorParameters()

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
            # Filter markers within the desired ID range (0-8)
            valid_markers = [(corners, marker_ID) for corners, marker_ID in zip(
                marker_corners, marker_IDs) if marker_ID[0] in range(9)]

            if valid_markers:
                # Assuming only one marker is detected
                corners, ids = valid_markers[0]
                rVec, tVec, _ = aruco.estimatePoseSingleMarkers(
                    [corners], MARKER_SIZE, cam_mat, dist_coef)
                i = 0

                distance = np.sqrt(tVec[i][0][2] ** 2 +
                                   tVec[i][0][0] ** 2 + tVec[i][0][1] ** 2)

                # Check if the distance is less than 70cm and ID is within 0-8
                if 0 <= ids[0] <= 8 and distance < 70:
                    cv.polylines(
                        gray_frame, [corners.astype(
                            np.int32)], True, (0, 255, 255), 4, cv.LINE_AA
                    )
                    corners = corners.reshape(4, 2)
                    corners = corners.astype(int)
                    top_right = corners[0].ravel()
                    bottom_right = corners[2].ravel()

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

                    # Output the detected marker information
                    # print("Detected marker: id={} distance={:.2f}".format(
                    #     ids[0], distance))

                    return ids[0], distance

        cv.imshow("gray_frame", gray_frame)
        key = cv.waitKey(1)

        if key == ord("q"):
            break

    cv.destroyAllWindows()


def calculate_speed(diameter_mm, rpm):
    # Calculate the circumference in millimeters
    circumference_mm = math.pi * diameter_mm
    # Convert speed to millimeters per second
    speed_mm_per_second = (rpm * circumference_mm)/60
    return speed_mm_per_second


def calculate_runTime(speed_mm_per_second, marker_distance):
    dueRunTime = marker_distance/speed_mm_per_second
    return dueRunTime


def motorController(marker_distance, marker_id, speedMain):
    wheel_diameter_mm = 44
    wheel_rpm = 300

    # Calculate the speed
    speed_mm_per_second = calculate_speed(wheel_diameter_mm, wheel_rpm)
    OnTime = calculate_runTime(speed_mm_per_second, marker_distance)

    print("Moving Speed {}mm/s,   On Time={:.2f}".format(
        speed_mm_per_second, OnTime))

    # Motor control based on marker distance
    if marker_distance is not None and marker_distance < 700:
        forwardDrive(speedMain, speedMain)

        # Keep moving forward for 'n' seconds
        start_time = time()
        while (time() - start_time) < OnTime:
            pass  # You can add additional processing here if needed

        # Stop the motors after moving forward for 'n' seconds
        forceBrake()
        allStop()


def main(stdscr):
    # Main speed
    speedMain = 0.5
    rotation_speed = 0.5
    # Define a threshold for heading proximity
    threshold = 5

    # Capture the starting heading
    start_heading = get_heading()

    picam2 = Picamera2()
    picam2.preview_configuration.main.size = (800, 800)
    picam2.preview_configuration.main.format = "RGB888"
    picam2.preview_configuration.align()
    picam2.configure("preview")
    picam2.start()

    while True:
        marker_id, marker_distance_in_cm = detect_aruco_marker(picam2)
        marker_distance = marker_distance_in_cm*10
        double_beep(0.1)

        motorController(marker_distance, marker_id, speedMain)

        print("Detected marker: id={} distance={:.2f}".format(
            marker_id, marker_distance))

        if marker_id == 1:
            point_east(rotation_speed, threshold, start_heading)
        elif marker_id == 2:
            point_south_east(rotation_speed, threshold, start_heading)
        elif marker_id == 3:
            point_south(rotation_speed, threshold, start_heading)
        elif marker_id == 4:
            point_south_west(rotation_speed, threshold, start_heading)
        elif marker_id == 5:
            point_west(rotation_speed, threshold, start_heading)
        elif marker_id == 6:
            point_north_west(rotation_speed, threshold, start_heading)
        elif marker_id == 7:
            point_north(rotation_speed, threshold, start_heading)
        elif marker_id == 8:
            point_north_east(rotation_speed, threshold, start_heading)
        else:
            break

    tri_beep(0.2)


if __name__ == "__main__":
    try:
        curses.wrapper(main)
    finally:
        cleanup_gpio()
