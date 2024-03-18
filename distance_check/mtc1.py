import RPi.GPIO as GPIO
from gpiozero import PWMOutputDevice, DigitalOutputDevice
from time import time, sleep
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
prev_time = time()

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


def main(win):
    global prev_time
    win.clear()
    win.keypad(1)

    # Main speed
    speedMain = 0.5
    # Adjust these values for the desired speeds
    forward_speed_left = speedMain
    forward_speed_right = speedMain
    reverse_speed_left = speedMain
    reverse_speed_right = speedMain
    turn_left_speed_left = speedMain
    turn_left_speed_right = speedMain
    turn_right_speed_left = speedMain
    turn_right_speed_right = speedMain

    while True:
        current_time = time()
        time_elapsed = current_time - prev_time

        key = win.getch()
        win.clear()

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


# Run the main function with GPIO cleanup on exit
if __name__ == "__main__":
    try:
        curses.wrapper(main)
    finally:
        cleanup_gpio()