import py_qmc5883l
import time
sensor = py_qmc5883l.QMC5883L()
sensor.calibration = [[1.0256545432028572, -0.013013085106172594, 289.7145364429366],
                      [-0.01301308510617258, 1.0066007951356402, 879.0513837979042], [0.0, 0.0, 1.0]]
while True:
    l = sensor.get_bearing()
    print(l)
    time.sleep(0.001)
