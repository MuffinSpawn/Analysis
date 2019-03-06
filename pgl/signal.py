import math
import numpy as np  # NumPy (multidimensional arrays, linear algebra, ...)
import numpy.linalg as linalg
import scipy.signal as sig
import scipy.stats as stats
import pgl.cluster as cluster
import pgl.curve as curve
import multiprocessing as mp
import matplotlib.pyplot as plt
import peakutils

# Compute the sum of the squared residuals between two equal-length signals.
def ssr(signal1, signal2):
  ssr_sum = 0.0
  for y1,y2 in zip(signal1,signal2):
    ssr_sum += (y1-y2)**2
  return ssr_sum

# Take two arrays of N sets of M signals, compute the total SSR for each pair
# of signal sets formed from the two arrays, and generate a matrix of all the
# total SSR values. This will, of course, be a symmetric matrix with zeros on
# the diagonal.
def ssr_matrix(signals1, signals2):
  signals_shape = np.shape(signals1)
  matrix = np.zeros((signals_shape[0], signals_shape[0]))
  for i,ys_array1 in enumerate(signals1):
    for j,ys_array2 in enumerate(signals2):
      total = 0.0
      for ys1,ys2 in zip(ys_array1,ys_array2):
        total += ssr(ys1,ys2)
      matrix[i,j] = total
      matrix[j,i] = total
  return matrix

def normalize(signals):
  for i,signal in enumerate(signals):
    peak = np.max(np.abs(signal))
    signals[i] = signal / peak
  return signals

def ac_contrib_index(mic_coordinates, test_coordinates,
                     thickness, i, j, n0, thetas, dt, settling_time):
  # - Calculate the expected difference in TOA
  xi = mic_coordinates[i,0]
  yi = mic_coordinates[i,1]
  dxi = xi-test_coordinates[0]
  dyi = yi-test_coordinates[1]
  di = math.sqrt(dxi*dxi+dyi*dyi+thickness*thickness)
  #v_i = thetas[0] +  thetas[1] * di
  v_i = 0
  for order,theta in enumerate(thetas):
    v_i += theta*di**order
  ti = di / v_i + settling_time*i

  xj = mic_coordinates[j,0]
  yj = mic_coordinates[j,1]
  dxj = xj-test_coordinates[0]
  dyj = yj-test_coordinates[1]
  dj = math.sqrt(dxj*dxj+dyj*dyj+thickness*thickness)
  #v_j = thetas[0] +  thetas[1] * dj
  v_j = 0
  for order,theta in enumerate(thetas):
    v_j += theta*dj**order
  tj = dj / v_j + settling_time*j

  tij = tj - ti
  return int(round(n0 - tij / dt - 1))


def accumulated_correlation(signals, dt, mic_coordinates, radius, thickness,
                            thetas, grid_size=10, settling_time=0, octant=-1):
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
  radius: scalar
    The inner radius of the cavity
  thickness: scaler
    The thickness of the end plate of the cavity
  v_p: scalar
    The speed of sound in the cavity material
  grid_size: int, optional
    The number of vertical and horizontal test points (default 10)
  settling_time: scalar
    The time it takes for the DAQ digitizer to read one channel

  Returns
  -------
  c : list
    A two-element list containing the x and y coordinates
  """
  """
  signals = np.copy(signals)
  for y in signals:
    peaks = peakutils.indexes(y, thres=0.1, min_dist=3)
    y[peaks[-1]-12:] = 0.0
  """
  #signals = np.abs(signals)
  #signals = signals**2
  """
  max_indicies = np.argmax(signals, axis=1)
  peak_amplitudes = signals[range(len(max_indicies)), max_indicies]
  peak_amplitude_sum = np.sum(peak_amplitudes)
  fractional_amplitudes = peak_amplitudes / peak_amplitude_sum
  closest_index = np.argmin(max_indicies)
  max_index = max_indicies[closest_index]
  mic_biases = []
  cov = np.array([[radius, 0],[0, radius]])
  for mic_index in range(len(signals)):
    mean = mic_coordinates[mic_index]
    mic_biases.append(stats.multivariate_normal(mean, cov=cov))
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

  # Determine coordinates of the test grid
  quadrant = -1
  if octant >=0:
    quadrant = int(octant / 2)
  if quadrant >= 0:
    grid_size /= 2
    if (quadrant == 0) or (quadrant == 3):
      xs = np.linspace(0, radius, num=grid_size)
    else:
      xs = np.linspace(0, -radius, num=grid_size)
    if (quadrant == 0) or (quadrant == 1):
      ys = np.linspace(0, radius, num=grid_size)
    else:
      ys = np.linspace(0, -radius, num=grid_size)
  else:
    xs = np.linspace(-radius, radius, num=grid_size)
    ys = np.linspace(-radius, radius, num=grid_size)

  # - Create a zero matrix the size of the test point grid (sum matrix)
  sums = np.zeros((grid_size, grid_size))

  n0 = len(signals[0])
  ijs = []
  for i,signal_i in enumerate(signals):
    for j,signal_j in enumerate(signals[i+1:]):
      # Note: j -> j+i+1 because of the loop optimization
      ijs.append([i, j+i+1])
  ijs = np.array(ijs)

  if np.any(octant == np.array([0,1,4,5])):
    # octants 0,1,4,5
    constraint_slope = float(mic_coordinates[0,1]) / mic_coordinates[0,0]
  else:
    # octants 2,3,6,7
    constraint_slope = float(mic_coordinates[1,1]) / mic_coordinates[1,0]
  """
  print 'Constraint Slope:', constraint_slope
  print 'Quadrant:', quadrant
  print 'xs:', xs
  print 'ys:', ys
  """
  # The math in the inner loop takes O(1) time
  # The inner two loops take O(N^2) time (N=# microphones)
  # The outer two loops take O(M^2) time if we assume equal sized
  # horizontal and vertical grids with M rows and columns
  # Together this is O(M^2*N^2) time
  #
  # - For each test point...
  #print 'xs = {%.2f, %.2f}' % (xs[0], xs[-1])
  for a,x in enumerate(xs):
    if (quadrant >= 0):
      max_y = math.sqrt(radius**2 - x**2)
      dy = radius / (grid_size-1)
      max_b = int(round(max_y / dy))

    else:
      min_b = 0
      max_b = len(ys)
    #print 'ys = {%.2f, %.2f}' % (ys[0], ys[max_b-1])
    for b,y in enumerate(ys[:max_b]):
    #for b,y in enumerate(ys):
      # - For each pair of microphones...
      for index,ij in enumerate(ijs):
        contrib_index = -1

        if (x**2 + y**2) <= (radius**2) and\
           ((octant == 0 and y <= constraint_slope*x and x >= y/constraint_slope) or\
           (octant == 1 and y >= constraint_slope*x and x <= y/constraint_slope) or\
           (octant == 2 and y >= constraint_slope*x and x >= y/constraint_slope) or\
           (octant == 3 and y <= constraint_slope*x and x <= y/constraint_slope) or\
           (octant == 4 and y >= constraint_slope*x and x <= y/constraint_slope) or\
           (octant == 5 and y <= constraint_slope*x and x >= y/constraint_slope) or\
           (octant == 6 and y <= constraint_slope*x and x <= y/constraint_slope) or\
           (octant == 7 and y >= constraint_slope*x and x >= y/constraint_slope) or\
           (octant < 0)):
          #print 'r=', x**2 + y**2
          contrib_index = ac_contrib_index(mic_coordinates, [x, y], thickness,
                                           ij[0], ij[1], n0, thetas, dt,
                                           settling_time)
        if contrib_index >= 0 and contrib_index < lag_matrix.shape[2]:
          sums[a,b] += lag_matrix[ij[0],ij[1],contrib_index]

  """
  min_indicies = np.unravel_index([np.argmin(sums)], np.shape(sums))
  min_coordinates = [xs[min_indicies[0][0]], ys[min_indicies[1][0]]]

  bias_matrix = np.ones(sums.shape)
  for mic_index,mic_bias in enumerate(mic_biases):
    X,Y = np.meshgrid(xs, ys)
    xy = np.array(zip(X.flatten(), Y.flatten()))
    bias_matrix += (mic_bias.pdf(xy).reshape(sums.shape).transpose()\
                   * fractional_amplitudes[mic_index])
                   #* math.exp(fractional_amplitudes[mic_index]))
  min_bias = stats.multivariate_normal(mean=min_coordinates, cov=[[radius,0],[0,radius]])
  X,Y = np.meshgrid(xs, ys)
  xy = np.array(zip(X.flatten(), Y.flatten()))
  bias_matrix += min_bias.pdf(xy).reshape(sums.shape).transpose()
  bias_matrix / np.max(bias_matrix)
  sums = sums * bias_matrix
  """

  # - Use the max sum matrix element to calculate the most likely source point
  max_indicies = np.unravel_index([np.argmax(sums)], np.shape(sums))
  coordinates = [xs[max_indicies[0][0]], ys[max_indicies[1][0]]]
  #if coordinates[0]**2 + coordinates[1]**2 > radius**2:
    #coordinates = [0.0,0.0]
  return coordinates

def peaks(xs, ys):
    derivative = np.gradient(ys)
    #print derivative
    last_deriv = 0.0
    indicies = []
    for index, deriv in enumerate(derivative):
        if ((last_deriv >= 0.0) and (deriv < 0.0)):
            if ((index > 0) and (ys[index-1] > ys[index])):
                indicies.append(index-1)
            else:
                indicies.append(index)
        last_deriv = deriv
    ysxs = zip(map((lambda i: xs[i]), indicies), map((lambda i: ys[i]), indicies))
    sorted_ysxs = sorted(ysxs, key=lambda elem: elem[1], reverse=True)
    return np.array(list(map((lambda elem: elem[0]), sorted_ysxs)))

def spectra(times, signals, padlen=0):
  dt = times[1] - times[0]
  if padlen > 0:
    signal_padding = np.zeros((np.shape(signals)[0], padlen))
    signals = np.hstack((signal_padding, signals, signal_padding))
    time_padding = np.arange(times[-1]+dt, times[-1]+dt+padlen*dt, dt)
    times = np.hstack((times, time_padding))
  spectrum_length = round(np.shape(signals)[1]/2)
  frequencies = np.zeros((spectrum_length))
  magnitudes = np.zeros((np.shape(signals)[0], spectrum_length))
  phases = np.zeros((np.shape(signals)[0], spectrum_length))
  for index,signal in enumerate(signals):
    frequency_spectrum = np.fft.fft(signal)[:spectrum_length]
    magnitudes[index] += np.sqrt(  np.real(frequency_spectrum)**2 \
                                 + np.imag(frequency_spectrum)**2)
    frequencies = np.fft.fftfreq(
      frequency_spectrum.size*2, d=dt)[:spectrum_length]
    phases[index] = np.arctan2(np.imag(frequency_spectrum),
                               np.real(frequency_spectrum))
  return (frequencies, magnitudes, phases)


def ricker_center_freq(dt):
  # Get the frequency spectrum of the Ricker wavelet
  # with width 1 in units of samples
  wavelet = sig.ricker(1e3, 1) # arbitrary num. of points, not too big or small
  frequency_spectrum = np.fft.fft(wavelet)[:int(round(wavelet.size/2))]
  spectrum_magnitude = np.sqrt(  np.real(frequency_spectrum)**2
                               + np.imag(frequency_spectrum)**2)
  frequencies = np.fft.fftfreq(frequency_spectrum.size*2, d=1)\
                  [:int(round(wavelet.size/2))]
  # Empirically, fc/fp = 1.186; where fp is the peak FFT frequency and
  # fc is the desired center frequency.
  #   - Divide by the sample width to get units of Hz
  return 1.186 * frequencies[np.argmax(spectrum_magnitude)] / dt

def translated_ricker_wavelet(times, scale, t):
  wavelet = sig.ricker(len(times), scale)
  dt = times[0] - times[1]
  zero_index = times.size / 2
  center_index = np.where(np.abs(times-t) < 1.0e-9)[0][0]
  shift = center_index - zero_index
  return np.roll(wavelet, shift)
