<<<<<<< Updated upstream
import py_qmc5883l
import time
sensor = py_qmc5883l.QMC5883L()
sensor.calibration = [[1.0256545432028572, -0.013013085106172594, 289.7145364429366],
                      [-0.01301308510617258, 1.0066007951356402, 879.0513837979042], [0.0, 0.0, 1.0]]
while True:
    l = sensor.get_bearing()
    print(l)
    time.sleep(0.001)
=======
import os
import sys
import time
import smbus
import numpy as np

from imusensor.MPU9250 import MPU9250
from imusensor.filters import kalman

# Get the directory of the script
script_dir = os.path.dirname(os.path.realpath(__file__))
calib_file_path = os.path.join(script_dir, "new_usk_calib.json")

address = 0x68
bus = smbus.SMBus(1)
imu = MPU9250.MPU9250(bus, address)
imu.begin()
# imu.caliberateAccelerometer()
# print ("Acceleration calib successful")
# imu.caliberateMag()
# print ("Mag calib successful")
# or load your caliberation file


# imu.loadCalibDataFromFile(calib_file_path)

sensorfusion = kalman.Kalman()

imu.readSensor()
imu.computeOrientation()
sensorfusion.roll = imu.roll
sensorfusion.pitch = imu.pitch
sensorfusion.yaw = imu.yaw

count = 0
currTime = time.time()

while True:
    imu.readSensor()
    imu.computeOrientation()
    newTime = time.time()
    dt = newTime - currTime
    currTime = newTime

    sensorfusion.computeAndUpdateRollPitchYaw(imu.AccelVals[0], imu.AccelVals[1], imu.AccelVals[2], imu.GyroVals[0], imu.GyroVals[1], imu.GyroVals[2],
                                              imu.MagVals[0], imu.MagVals[1], imu.MagVals[2], dt)



    if sensorfusion.yaw < 0:
        heading = (360 - abs(sensorfusion.yaw) ) 
    else:
        heading = (abs(sensorfusion.yaw) )

    
    refNorth = heading
    


    print("Heading:{1} RefNorth:{2} ".format(
        sensorfusion.yaw,  heading, refNorth ))

    time.sleep(0.01)
>>>>>>> Stashed changes
