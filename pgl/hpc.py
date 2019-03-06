import math
import numpy as np
import scipy.signal as sig
import pgl.tree as tree
import pgl.signal as psig

def localize_dlhpc(times, signals, rotation):
  # 0 deg, 120 deg, 90 deg
  mic_coordinates = np.array([zip([-10.9, 4.2, 6.7], [1.4, -10.2, 8.7]),
                              zip([4.2, 6.7, -10.9], [-10.2, 8.7, 1.4]),
                              zip([-1.4, 10.2, -8.7], [-10.9, 4.2, 6.7])])
  dt=5.0e-6
  radius = 14.22  # cm
  thickness = 1.37  # cm
  v = 4.4e5  # cm/s
  spark_location = np.zeros((2))
  damped_signals = np.zeros((np.shape(signals)[0], np.shape(signals)[1]))
  for index,signal in enumerate(signals):
    damped_signals[index] = [signals[index,x]*(math.exp(1-times[x]/2.0e-5)) for x in range(np.shape(signals)[1])]
  return psig.accumulated_correlation(damped_signals, dt, mic_coordinates[rotation],
                                      radius, thickness, v, v, grid_size=100)
