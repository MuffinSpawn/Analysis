import math
import numpy as np
import scipy.signal as sig
import pgl.tree as tree
import pgl.signal as psig
import pgl.comsol as comsol

def order_by_width(signals, max_width=10):
    M,N = np.shape(signals)
    widths = np.linspace(1, max_width, num=N)
    peak_width_tree = tree.BinarySearchTree()
    for index,signal in enumerate(signals):
        cwtmatr = sig.cwt(signals[index], sig.ricker, widths)
        min_indicies = np.unravel_index(cwtmatr.argmin(), cwtmatr.shape)
        node = tree.BinaryTreeNode(index, widths[min_indicies[0]])
        peak_width_tree.insert(node)
    return peak_width_tree.walk_in_order()

def order_by_time(times, signals, max_width=10, cwt_max=True):
    M,N = np.shape(signals)
    widths = np.linspace(1, max_width, num=N)
    mic_tree = tree.BinarySearchTree()
    for index,signal in enumerate(signals):
        cwtmatr = sig.cwt(signals[index], sig.ricker, widths)
        if cwt_max:
          indicies = np.unravel_index(cwtmatr.argmax(), cwtmatr.shape)
        else:
          indicies = np.unravel_index(cwtmatr.argmin(), cwtmatr.shape)
        node = tree.BinaryTreeNode(index, times[indicies[1]])
        mic_tree.insert(node)
    return mic_tree.walk_in_order()

def accumulated_correlation(signals, dt, coordinates, x_lims, y_lims, z_lims,
                            v_p, grid_size=10):
  """
  Estimate the location of an acoustic event from multiple microphone signals

  The Accumulated Correlation algorithm estimates the source location of a
  signal that arrives at different times at multiple sensors. It starts by
  calculating the cross-correlation for each pair of microphones signals. For
  each test grid point, the expected time delay is calculated for each
  microphone. Then for each unique signal pair the difference in the expected
  time delay is used as an index into the cross correlation vectors. The
  value in the cross correlation vector is added to a running sum for the
  current test grid point. Finally, the test grid point with the largest sum
  is taken as the most likely source location of the signal.

  Parameters
  ----------
  signals : numpy.ndarray
    An array of time-domain signals
  dt : scalar
    The amount of time between each signal sample
  coordinates: numpy.ndarray
    An array of microphone coordinates (2D array with dimensions N x 2)
  x_lims: numpy.array
    The x-axis limits of the search grid
  y_lims: numpy.array
    The y-axis limits of the search grid
  z_lims: numpy.array
    The z-axis limits of the search grid
  v_p: scalar
    The speed of sound in the cavity material
  grid_size: int, optional
    The number of vertical and horizontal test points (default 10)

  Returns
  -------
  c : list
    A two-element list containing the x and y coordinates
  """
  # The cross-correlation takes O(n) time, where n is the length of the signals
  # These loops take O(N^2) time, where N is the number of signals
  # For constant N, then, increasing the signal size linearly increases the
  # running time
  #
  # - Calculate the lag matrix (skip auto correlations since they aren't used)
  lag_matrix = np.zeros((len(signals), len(signals), len(signals[0])*2-1))
  for i,signal_i in enumerate(signals):
    for j,signal_j in enumerate(signals[i+1:]):
      lag_matrix[i, j+i+1] = sig.correlate(signal_i, signal_j)
      lag_matrix[j+i+1, i] = lag_matrix[i, j+i+1]

  # - Create a zero matrix the size of the test point grid (sum matrix)
  sums = np.zeros((grid_size, grid_size, grid_size))
  xs = np.linspace(x_lims[0], x_lims[1], num=grid_size)
  ys = np.linspace(y_lims[0], y_lims[1], num=grid_size)
  zs = np.linspace(z_lims[0], z_lims[1], num=grid_size)
  # The math in the inner loop takes O(1) time
  # The inner two loops take O(N^2) time
  # The outer two loops take O(M^2) time if we assume equal sized horizontal
  # and vertical grids with M rows and columns
  # Together this is O(M^2*N^2) time
  #
  # - For each test point...
  for a,x in enumerate(xs):
    for b,y in enumerate(ys):
      for c,z in enumerate(zs):
        # - For each pair of microphones...
        for i,signal_i in enumerate(signals):
          for j,signal_j in enumerate(signals[i+1:]):
            # - Calculate the expected difference in TOA
            xi = coordinates[i,0]
            yi = coordinates[i,1]
            zi = coordinates[i,2]
            dxi = xi-x
            dyi = yi-y
            dzi = zi-z
            di = math.sqrt(dxi*dxi+dyi*dyi+dzi*dzi)
            ti = di / v_p

            # Note: j -> j+i+1 because of the loop optimization
            xj = coordinates[j+i+1,0]
            yj = coordinates[j+i+1,1]
            zj = coordinates[j+i+1,2]
            dxj = xj-x
            dyj = yj-y
            dzj = zj-z
            dj = math.sqrt(dxj*dxj+dyj*dyj+dzj*dzj)
            tj = dj / v_p

            tij = tj - ti
            n0 = len(signals[0])
            k = int(round(n0 - tij / dt - 1))

            # - Add the appropriate lag matrix value for the given TOA delta to the
            #   sum matrix
            sums[a,b,c] += lag_matrix[i,j+i+1,k]
  # - Use the max sum matrix element to calculate the most likely source point
  max_indicies = np.unravel_index([np.argmax(sums)], np.shape(sums))
  return [xs[max_indicies[0][0]], ys[max_indicies[1][0]], zs[max_indicies[2][0]]]

def accumulated_correlation_2D(signals, dt, coordinates, x_lims, y_lims, z,
                            v_p, grid_size=10):
  """
  Estimate the location of an acoustic event from multiple microphone signals

  The Accumulated Correlation algorithm estimates the source location of a
  signal that arrives at different times at multiple sensors. It starts by
  calculating the cross-correlation for each pair of microphones signals. For
  each test grid point, the expected time delay is calculated for each
  microphone. Then for each unique signal pair the difference in the expected
  time delay is used as an index into the cross correlation vectors. The
  value in the cross correlation vector is added to a running sum for the
  current test grid point. Finally, the test grid point with the largest sum
  is taken as the most likely source location of the signal.

  Parameters
  ----------
  signals : numpy.ndarray
    An array of time-domain signals
  dt : scalar
    The amount of time between each signal sample
  coordinates: numpy.ndarray
    An array of microphone coordinates (2D array with dimensions N x 2)
  x_lims: numpy.array
    The x-axis limits of the search grid
  y_lims: numpy.array
    The y-axis limits of the search grid
  z: scalar
    z position
  v_p: scalar
    The speed of sound in the cavity material
  grid_size: int, optional
    The number of vertical and horizontal test points (default 10)

  Returns
  -------
  c : list
    A two-element list containing the x and y coordinates
  """
  # The cross-correlation takes O(n) time, where n is the length of the signals
  # These loops take O(N^2) time, where N is the number of signals
  # For constant N, then, increasing the signal size linearly increases the
  # running time
  #
  # - Calculate the lag matrix (skip auto correlations since they aren't used)
  lag_matrix = np.zeros((len(signals), len(signals), len(signals[0])*2-1))
  for i,signal_i in enumerate(signals):
    for j,signal_j in enumerate(signals[i+1:]):
      lag_matrix[i, j+i+1] = sig.correlate(signal_i, signal_j)
      lag_matrix[j+i+1, i] = lag_matrix[i, j+i+1]

  # - Create a zero matrix the size of the test point grid (sum matrix)
  sums = np.zeros((grid_size, grid_size))
  xs = np.linspace(x_lims[0], x_lims[1], num=grid_size)
  ys = np.linspace(y_lims[0], y_lims[1], num=grid_size)
  # The math in the inner loop takes O(1) time
  # The inner two loops take O(N^2) time
  # The outer two loops take O(M^2) time if we assume equal sized horizontal
  # and vertical grids with M rows and columns
  # Together this is O(M^2*N^2) time
  #
  # - For each test point...
  for a,x in enumerate(xs):
    for b,y in enumerate(ys):
      # - For each pair of microphones...
      for i,signal_i in enumerate(signals):
        for j,signal_j in enumerate(signals[i+1:]):
          # - Calculate the expected difference in TOA
          xi = coordinates[i,0]
          yi = coordinates[i,1]
          dxi = xi-x
          dyi = yi-y
          di = math.sqrt(dxi*dxi+dyi*dyi+z*z)
          ti = di / v_p

          # Note: j -> j+i+1 because of the loop optimization
          xj = coordinates[j+i+1,0]
          yj = coordinates[j+i+1,1]
          dxj = xj-x
          dyj = yj-y
          dj = math.sqrt(dxj*dxj+dyj*dyj+z*z)
          tj = dj / v_p

          tij = tj - ti
          n0 = len(signals[0])
          k = int(round(n0 - tij / dt - 1))

          # - Add the appropriate lag matrix value for the given TOA delta to the
          #   sum matrix
          sums[a,b] += lag_matrix[i,j+i+1,k]
  # - Use the max sum matrix element to calculate the most likely source point
  max_indicies = np.unravel_index([np.argmax(sums)], np.shape(sums))
  return [xs[max_indicies[0][0]], ys[max_indicies[1][0]]]

def localize_spark(times, signals, dt=100.0e-6, grid_size=10):
  mic_coordinates = np.array(zip([250,250,250,500,500,500,750,750,750],
                                 [125,250,375,125,250,375,125,250,375],
                                 [510,510,510,510,510,510,510,510,510]))
  x_lims = np.array([[0,375],[0,375],[0,375],[375,625],[375,625],[375,625],[625,1000],[625,1000],[625,1000]], dtype='float64')
  y_lims = np.array([[0,187.5],[187.5,312.5],[312.5,500],[0,187.5],[187.5,312.5],[312.5,500],[0,187.5],[187.5,312.5],[312.5,500]])
  z_lims = np.array([0, 500], dtype='float64')
  length = 1000.0 # mm
  width = 500.0 # mm
  height = 500.0 # mm
  v_air = 3.4e5  # mm/s
  spark_location = np.zeros((2))
  damped_signals = np.zeros((np.shape(signals)[0], np.shape(signals)[1]))
  for index,signal in enumerate(signals):
    damped_signals[index] = [signals[index,x]*(math.exp(1-times[x]/2.0e-5)) for x in range(np.shape(signals)[1])]
  ordered_mics = order_by_time(times, damped_signals,
                               max_width=20, cwt_max=False)
  closest_mic = ordered_mics[0].id()
  return accumulated_correlation(damped_signals, dt, mic_coordinates,
                                 x_lims[closest_mic], y_lims[closest_mic],
                                 z_lims, v_air, grid_size=grid_size)

def localize_spark_2D(times, signals, z, dt=100.0e-6, grid_size=10):
  mic_coordinates = np.array(zip([250,250,250,500,500,500,750,750,750],
                                 [125,250,375,125,250,375,125,250,375],
                                 [510,510,510,510,510,510,510,510,510]))
  x_lims = np.array([[0,375],[0,375],[0,375],[375,625],[375,625],[375,625],[625,1000],[625,1000],[625,1000]], dtype='float64')
  y_lims = np.array([[0,187.5],[187.5,312.5],[312.5,500],[0,187.5],[187.5,312.5],[312.5,500],[0,187.5],[187.5,312.5],[312.5,500]])
  length = 1000.0 # mm
  width = 500.0 # mm
  height = 500.0 # mm
  v_air = 3.4e5  # mm/s
  spark_location = np.zeros((2))
  damped_signals = np.zeros((np.shape(signals)[0], np.shape(signals)[1]))
  for index,signal in enumerate(signals):
    damped_signals[index] = [signals[index,x]*(math.exp(1-times[x]/2.0e-5)) for x in range(np.shape(signals)[1])]
  ordered_mics = order_by_time(times, damped_signals,
                               max_width=20, cwt_max=False)
  closest_mic = ordered_mics[0].id()
  return accumulated_correlation_2D(damped_signals, dt, mic_coordinates,
                                 x_lims[closest_mic], y_lims[closest_mic],
                                 z, v_air, grid_size=grid_size)
