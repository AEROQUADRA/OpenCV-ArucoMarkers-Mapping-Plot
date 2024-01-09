import pyautogui
import random
import string
import time

# Set the minimum and maximum duration between key presses (in seconds)
min_delay = 2
max_delay = 60

# Infinite loop to simulate random key presses with random pauses
while True:
    # Generate a random letter from the alphabet
    random_letter = random.choice(string.ascii_lowercase)

    # Simulate key press
    pyautogui.press(random_letter)

    # Generate a random duration for the pause between key presses
    pause_duration = random.uniform(min_delay, max_delay)

    # Wait for the random duration
    time.sleep(pause_duration)
