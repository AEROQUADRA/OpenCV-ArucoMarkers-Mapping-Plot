import RPi.GPIO as GPIO
import time

# Set up GPIO using BCM numbering
GPIO.setmode(GPIO.BCM)

# Set GPIO 16 as output
GPIO.setup(16, GPIO.OUT)

try:
    # Turn GPIO 16 high
    GPIO.output(16, GPIO.HIGH)
    print("GPIO 16 set high")
    time.sleep(1)

finally:
    # Clean up GPIO settings
    GPIO.cleanup()
