import py_qmc5883l
import time
import curses
import RPi.GPIO as GPIO
from gpiozero import PWMOutputDevice, DigitalOutputDevice

sensor = py_qmc5883l.QMC5883L()
sensor.calibration = [[1.0256545432028572, -0.013013085106172594, 289.7145364429366],
                      [-0.01301308510617258, 1.0066007951356402, 879.0513837979042], [0.0, 0.0, 1.0]]

# Motor A, Left Side GPIO CONSTANTS
PWM_DRIVE_LEFT = 12  # ENA - H-Bridge enable pin
FORWARD_LEFT_PIN = 26  # IN1 - Forward Drive
REVERSE_LEFT_PIN = 19  # IN2 - Reverse Drive

# Motor B, Right Side GPIO CONSTANTS
PWM_DRIVE_RIGHT = 13  # ENB - H-Bridge enable pin
FORWARD_RIGHT_PIN = 21  # IN1 - Forward Drive
REVERSE_RIGHT_PIN = 20  # IN2 - Reverse Drive

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
    time.sleep(0.03)

    # Quick backward pulse
    forwardLeft.off()
    reverseLeft.on()
    forwardRight.off()
    reverseRight.on()
    driveLeft.value = 0.5
    driveRight.value = 0.5
    time.sleep(0.03)

    # Quick forward pulse
    forwardLeft.on()
    reverseLeft.off()
    forwardRight.on()
    reverseRight.off()
    driveLeft.value = 0.5
    driveRight.value = 0.5
    time.sleep(0.03)

    # Quick backward pulse
    forwardLeft.off()
    reverseLeft.on()
    forwardRight.off()
    reverseRight.on()
    driveLeft.value = 0.5
    driveRight.value = 0.5
    time.sleep(0.03)

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


def point_north(rotation_speed, threshold):
    desired_heading = 270
    point_direction(rotation_speed, desired_heading, threshold)


def point_south(rotation_speed, threshold):
    desired_heading = 90
    point_direction(rotation_speed, desired_heading, threshold)


# Define similar functions for other directions

def main(win):
    # Main speed
    speedMain = 0.5
    rotation_speed = 0.2
    # Define a threshold for heading proximity
    threshold = 5

    # Adjust these values for the desired speeds
    forward_speed_left = speedMain
    forward_speed_right = speedMain
    reverse_speed_left = speedMain
    reverse_speed_right = speedMain
    turn_speed = rotation_speed

    # Capture the starting heading
    start_heading = get_heading()

    # If start_heading is None, exit the function or handle the error accordingly
    if start_heading is None:
        print("Error: Unable to get the starting heading")
        return

    # Define a threshold for heading proximity
    threshold = 10

    point_north(rotation_speed, threshold)
    point_south(rotation_speed, threshold)

    return


if __name__ == "__main__":
    try:
        curses.wrapper(main)
    finally:
        cleanup_gpio()
