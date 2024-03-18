import py_qmc5883l
import time
import curses
import RPi.GPIO as GPIO
from gpiozero import PWMOutputDevice, DigitalOutputDevice
from time import time, sleep

sensor = py_qmc5883l.QMC5883L()
sensor.calibration = [[1.0256545432028572, -0.013013085106172594, 289.7145364429366],
                      [-0.01301308510617258, 1.0066007951356402, 879.0513837979042], [0.0, 0.0, 1.0]]


def get_heading():
    return sensor.get_bearing()


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
prev_time = time()

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


def forwardDrive(speed_left, speed_right, desired_heading):
    current_heading = get_heading()
    heading_difference = current_heading - desired_heading

    # Define adjustment factor based on heading difference
    adjustment_factor = 0.1

    # Calculate adjustment for each wheel based on heading difference
    # Positive heading_difference means the robot is deviating to the right
    # Negative heading_difference means the robot is deviating to the left
    left_adjustment = heading_difference * adjustment_factor
    right_adjustment = -heading_difference * adjustment_factor

    # Apply adjustment to motor speeds
    adjusted_speed_left = speed_left - left_adjustment
    adjusted_speed_right = speed_right - right_adjustment

    # Ensure motor speeds are within valid range (0 to 1)
    adjusted_speed_left = max(0, min(1, adjusted_speed_left))
    adjusted_speed_right = max(0, min(1, adjusted_speed_right))

    # Set motor speeds based on heading difference
    if heading_difference > 0:
        # Robot is deviating to the right, slow down left wheel
        forwardLeft.on()
        reverseLeft.off()
        forwardRight.on()
        reverseRight.off()
        driveLeft.value = adjusted_speed_left
        driveRight.value = speed_right
    elif heading_difference < 0:
        # Robot is deviating to the left, slow down right wheel
        forwardLeft.on()
        reverseLeft.off()
        forwardRight.on()
        reverseRight.off()
        driveLeft.value = speed_left
        driveRight.value = adjusted_speed_right
    else:
        # Robot is on desired heading, move forward at desired speeds
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


def main(win):
    global prev_time
    win.clear()
    win.keypad(1)

    # Main speed
    speedMain = 0.5

    # Capture the starting heading as the desired heading
    desired_heading = get_heading()  # Set desired heading as the starting heading

    # Adjust these values for the desired speeds
    forward_speed_left = speedMain
    forward_speed_right = speedMain
    reverse_speed_left = speedMain
    reverse_speed_right = speedMain
    turn_left_speed_left = speedMain
    turn_left_speed_right = speedMain
    turn_right_speed_left = speedMain
    turn_right_speed_right = speedMain

    spinLeft(speedMain, speedMain)
    start_time = time()
    while (time() - start_time) < 0.71:
        pass  # You can add additional processing here if needed

    # Stop the motors after moving forward for 'n' seconds
    forceBrake()

    while True:
        current_time = time()
        time_elapsed = current_time - prev_time

        key = win.getch()
        win.clear()

        if key == curses.KEY_UP:
            forwardDrive(forward_speed_left, forward_speed_right,
                         desired_heading)  # Pass desired heading here
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
            forceBrake()
            win.addstr("Stop")

        win.refresh()
        prev_time = current_time


# Run the main function with GPIO cleanup on exit
if __name__ == "__main__":
    try:
        curses.wrapper(main)
    finally:
        cleanup_gpio()
