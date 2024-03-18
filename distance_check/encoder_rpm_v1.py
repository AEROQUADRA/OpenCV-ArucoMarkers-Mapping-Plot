import time
import numpy as np
from gpiozero import RotaryEncoder
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Assigning parameter values
ppr = 300.8  # Pulses Per Revolution of the encoder
tstop = 20  # Loop execution duration (s)
tsample = 0.01  # Sampling period for code execution (s)
tdisp = 0.01  # Sampling period for values display (s)

# Encoder GPIO Pins
ENCODER_FL_C1 = 18
ENCODER_FL_C2 = 27
ENCODER_FR_C1 = 22
ENCODER_FR_C2 = 23
ENCODER_RL_C1 = 24
ENCODER_RL_C2 = 10
ENCODER_RR_C1 = 5
ENCODER_RR_C2 = 6

# Creating encoder objects using GPIO pins
encoder_fl = RotaryEncoder(ENCODER_FL_C1, ENCODER_FL_C2, max_steps=0)
encoder_fr = RotaryEncoder(ENCODER_FR_C1, ENCODER_FR_C2, max_steps=0)
encoder_rl = RotaryEncoder(ENCODER_RL_C1, ENCODER_RL_C2, max_steps=0)
encoder_rr = RotaryEncoder(ENCODER_RR_C1, ENCODER_RR_C2, max_steps=0)

# Initializing previous values and starting main clock
angleprev_fl = 0
angleprev_fr = 0
angleprev_rl = 0
angleprev_rr = 0
tprev = 0
tcurr = 0
tstart = time.perf_counter()

# Lists to store data for plotting
time_data = []
rpm_fl_data = []
rpm_fr_data = []
rpm_rl_data = []
rpm_rr_data = []

# Function to update the plot
def update_plot(frame):
    global tcurr, tprev, angleprev_fl, angleprev_fr, angleprev_rl, angleprev_rr

    # Getting current time (s)
    tcurr = time.perf_counter() - tstart

    # Getting angular position of the encoders (deg.)
    anglecurr_fl = 360 / ppr * encoder_fl.steps
    anglecurr_fr = 360 / ppr * encoder_fr.steps
    anglecurr_rl = 360 / ppr * encoder_rl.steps
    anglecurr_rr = 360 / ppr * encoder_rr.steps

    # Calculating change in angle and change in time
    dangle_fl = anglecurr_fl - angleprev_fl
    dangle_fr = anglecurr_fr - angleprev_fr
    dangle_rl = anglecurr_rl - angleprev_rl
    dangle_rr = anglecurr_rr - angleprev_rr
    dtime = tcurr - tprev

    # Calculating RPM
    rpm_fl = (dangle_fl / dtime) * (60 / ppr)
    rpm_fr = (dangle_fr / dtime) * (60 / ppr)
    rpm_rl = (dangle_rl / dtime) * (60 / ppr)
    rpm_rr = (dangle_rr / dtime) * (60 / ppr)

    # Appending data to lists for plotting
    time_data.append(tcurr)
    rpm_fl_data.append(rpm_fl)
    rpm_fr_data.append(rpm_fr)
    rpm_rl_data.append(rpm_rl)
    rpm_rr_data.append(rpm_rr)

    # Updating previous values
    angleprev_fl = anglecurr_fl
    angleprev_fr = anglecurr_fr
    angleprev_rl = anglecurr_rl
    angleprev_rr = anglecurr_rr
    tprev = tcurr

    # Plotting the data
    plt.clf()
    plt.plot(time_data, rpm_fl_data, label='FL RPM')
    plt.plot(time_data, rpm_fr_data, label='FR RPM')
    plt.plot(time_data, rpm_rl_data, label='RL RPM')
    plt.plot(time_data, rpm_rr_data, label='RR RPM')
    plt.title('Encoder RPM over Time')
    plt.xlabel('Time (seconds)')
    plt.ylabel('RPM')
    plt.legend()
    plt.grid(True)

# Running the animation
ani = FuncAnimation(plt.gcf(), update_plot, interval=tsample * 10, blit=False)

# Display the plot
plt.show()

# Closing GPIO pins when the plot window is closed
encoder_fl.close()
encoder_fr.close()
encoder_rl.close()
encoder_rr.close()
