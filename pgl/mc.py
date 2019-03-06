import math
import platform
import numpy
import numpy.linalg
import scipy.signal
import pgl.tree
import pgl.signal
import pgl.curve
import pgl.neural

class OctantDecisionTree:
    def __init__(self, node_map):
        self._root = None
        node = None
        for item in node_map:
            if self._root == None:
                self._root = pgl.tree.BinaryTreeNode(0, item)
                node = self._root
            else:
                id = 0
                if item >= 10:
                    id = 1
                elif item < 0:
                    id = -1
                if node._left == None:
                    node._left = pgl.tree.BinaryTreeNode(id, item)
                    node._left._parent = node
                    if (item >= 0) and (item < 10):
                        node = node._left
                elif node._right == None:
                    node._right = pgl.tree.BinaryTreeNode(id, item)
                    node._right._parent = node
                    if (item >= 0) and (item < 10):
                        node = node._right
                else:
                    while node._right != None:
                        node = node._parent
                    node._right = pgl.tree.BinaryTreeNode(id, item)
                    node._right._parent = node
                    if (item >= 0) and (item < 10):
                        node = node._right

    def get_octant(self, nodes):
        node = self._root
        index = 0
        while node._id == 0:
            if nodes[index]._id == node._value:
                node = node._left
                index += 1
            else:
                node = node._right
        return node._value - 10

    def _walk_in_order(self, node, nodes):
        if node._left != None:
            self._walk_in_order(node._left, nodes)
        nodes.append(node)
        if node._right != None:
            self._walk_in_order(node._right, nodes)
        return nodes

    def walk_in_order(self):
      return self._walk_in_order(self._root, [])

    def __repr__(self):
        nodes = self.walk_in_order()
        return str(nodes)

def order_by_width(signals, max_width=10):
    M,N = numpy.shape(signals)
    widths = numpy.linspace(1, max_width, num=N)
    peak_width_tree = pgl.tree.BinarySearchTree()
    for index,signal in enumerate(signals):
        cwtmatr = scipy.signal.cwt(signals[index], scipy.signal.ricker, widths)
        min_indicies = numpy.unravel_index(cwtmatr.argmin(), cwtmatr.shape)
        node = pgl.tree.BinaryTreeNode(index, widths[min_indicies[0]])
        peak_width_pgl.tree.insert(node)
    return peak_width_pgl.tree.walk_in_order()
"""
def order_by_time(times, signals, max_width=10, cwt_max=True):
    M,N = numpy.shape(signals)
    widths = numpy.linspace(1, max_width, num=N)
    mic_tree = pgl.tree.BinarySearchTree()
    for index,signal in enumerate(signals):
        cwtmatr = scipy.signal.cwt(signals[index], scipy.signal.ricker, widths)
        if cwt_max:
          indicies = numpy.unravel_index(cwtmatr.argmax(), cwtmatr.shape)
        else:
          indicies = numpy.unravel_index(cwtmatr.argmin(), cwtmatr.shape)
        node = pgl.tree.BinaryTreeNode(index, times[indicies[1]])
        mic_tree.insert(node)
    return mic_tree.walk_in_order()
"""
def order_by_time(times, signals, live=True):
  mic_tree = pgl.tree.BinarySearchTree()
  for index,signal in enumerate(signals):
    grad = numpy.gradient(signal)
    signs = numpy.sign(grad)
    signs[signs==0] = -1
    peak_indicies = numpy.where(numpy.diff(signs) < 0)
    peak_times = times[peak_indicies]
    peak_amplitudes = signal[peak_indicies]
    #print 'Peak Amplitudes:', peak_amplitudes
    max_peak_amplitude = peak_amplitudes.max()
    constrained_peak_indicies = None
    if live:
      # real data has noise before the first peak, so pick the first peak that
      # is above half the signal amplitude
      constrained_peak_indicies = peak_amplitudes > (max_peak_amplitude/2.0)
      #print 'Constrained Peak Indicies:', constrained_peak_indicies
      if numpy.all(numpy.logical_not(constrained_peak_indicies)):
        constrained_peak_indicies = [peak_amplitudes.argmax()]
    else:
      # since simulations have no noise, just pick the first peak we see
      constrained_peak_indicies = [peak_amplitudes.argmax()]
    node = pgl.tree.BinaryTreeNode(index,
                                   peak_times[constrained_peak_indicies][0])
    mic_tree.insert(node)
  return mic_tree.walk_in_order()

def order_by_distance(distances):
  mic_tree = pgl.tree.BinarySearchTree()
  for index,distance in enumerate(distances):
    node = pgl.tree.BinaryTreeNode(index, distance)
    mic_tree.insert(node)
  return mic_tree.walk_in_order()

def order_by_amplitude(times, signals, max_width=10, cwt_max=True):
  mic_tree = pgl.tree.BinarySearchTree()
  for index,signal in enumerate(signals):
    amplitude = numpy.max(numpy.abs(signal))
    print("amplitude:",amplitude)
    node = pgl.tree.BinaryTreeNode(index, 1.0/amplitude)
    mic_tree.insert(node)
  last_start = int(0)
  ids = [n.id() for n in mic_tree.walk_in_order()]
  print("IDs:",ids)

  M,N = numpy.shape(signals)
  widths = numpy.linspace(1, max_width, num=N)
  mic_tree = pgl.tree.BinarySearchTree()
  for id in ids:
    cwtmatr = scipy.signal.cwt(signals[id,last_start:], scipy.signal.ricker, widths)
    if cwt_max:
      indicies = numpy.unravel_index(cwtmatr.argmax(), cwtmatr.shape)
    else:
      indicies = numpy.unravel_index(cwtmatr.argmin(), cwtmatr.shape)
    if (last_start == 0):
      index = last_start + indicies[1]
    node = pgl.tree.BinaryTreeNode(id, times[index])
    mic_tree.insert(node)
  return mic_tree.walk_in_order()

def octant_from_net(signals):
  data_dir = "C:\\Users\\plane\\Desktop\\Data\\MC\\"
  if platform.system() == 'Linux':
      data_dir = "/home/lane/Data/MC/"
  weights_file = ''.join((data_dir, 'quadrant_weights.npy'))
  weights = numpy.load(weights_file)
  weights_shapes = numpy.array([w.shape for w in weights])
  layer_sizes = numpy.hstack((weights_shapes[0,1]-1, weights_shapes[:,0]))
  net = pgl.neural.Net(layer_sizes)
  net.set_weights(weights)

  x = numpy.hstack([signals.flatten(), signals.flatten()])
  y = net.forward_propagate(x, weights).flatten() >= 0.5
  indicies = numpy.argwhere(y)
  if len(indicies) > 0:
    quadrant = indicies[0,0]
  else:
    quadrant = 0
  octant = quadrant * 2
  return octant

def octant_trilateration(widths):
    # print 'Widths:', widths
    """
    node_map = [0, 1, 3, 10, -1, 3, 1, 11, -1, -1,
                1, 0, 2, 12, -1, 2, 0, 13, -1, -1,
                2, 1, 3, 14, -1, 3, 1, 15, -1, -1,
                3, 2, 0, 16, -1, 0, 2, 17, -1, -1, -1]
    """
    node_map = [0, 3, 10, 1, 11, -1,
                1, 0, 12, 2, 13, -1,
                2, 1, 14, 3, 15, -1,
                3, 2, 16, 0, 17, -1, -1]
    octants = OctantDecisionTree(node_map)
    #print octants

    return octants.get_octant(widths)

def trilaterateX(r1, r2, x1, x2, y12):
  x0 = x2 - x1
  return (r1**2 - r2**2 + x0**2) / (2*x0) + x1  # x

def trilaterateY(r1, r2, y1, y2, x12):
  y0 = y2 - y1
  return (r1**2 - r2**2 + y0**2) / (2*y0) + y1  # y

def trilaterateXY(r1, r2, x1, y1, x2, y2):
  print('(%.2f,%.2f), (%.2f,%.2f)' % (x1, y1, x2, y2))
  r12 = math.sqrt((x1-x2)**2 + (y1-y2)**2)
  a = (r1**2 - r2**2 + r12**2) / (2 * r12)
  theta = math.atan2(y2, x2)
  print('r1:', r1, 'r2:', r2, 'theta:', theta, 'a:', a)
  a_x = a * math.cos(theta) + x1
  a_y = a * math.sin(theta) + y1
  m = -a_x / a_y
  b = a_y + a_x**2 / a_y
  #print 'm:', m, 'b:', b
  print("a_x",a_x,"a_y:",a_y)
  return numpy.array([m, b])

def trilaterate(ts):
  mic_xs = [5, -5, -5, 5]
  mic_ys = [6, 6, -6, -6]
  #v_p = 4.76e5  # cm/ms
  v_p = 3.5e5  # cm/ms
  rs = ts * v_p
  line_xs = []
  line_ys = []
  ms = []
  bs = []
  index = int(0)
  for i,r1 in enumerate(rs[:-1]):
    for j,r2 in enumerate(rs[i+1:]):
      x1 = mic_xs[i]
      y1 = mic_ys[i]
      x2 = mic_xs[j+i+1]
      y2 = mic_ys[j+i+1]
      #print x1,y1,x2,y2
      if (x1 == x2):
        if (r1 == r2):
          line_ys.append(0.0)
        else:
          #lines[index,1] = trilaterateY(r1, r2, y1, y2, x1)
          line_ys.append(trilaterateY(r1, r2, y1, y2, x1))
          #print "x1==x2",line_ys[-1]
          #print "x1==x2", (r1,r2,y1,y2,x1)
          #print 'y:', ys[-1]
      elif (y1 == y2):
        if (r1 == r2):
          line_xs.append(0.0)
        else:
          #lines[index,0] = trilaterateX(r1, r2, x1, x2, y1)
          line_xs.append(trilaterateX(r1, r2, x1, x2, y1))
          #print 'x:', xs[-1]
          #print "y1==y2",line_xs[-1]
          #print "y1==y2",(r1,r2,x1,x2,y1)
      else:
        if (r1==r2):
          (m,b) = (1.0, 0.0)
        else:
          #lines[index,:] = trilaterateXY(r1, r2, x1, y1, x2, y2)
          (m, b) = trilaterateXY(r1, r2, x1, y1, x2, y2)
          #print 'm:', m, 'b:', b
        ms.append(m)
        bs.append(b)
        #print "else",(m,b)
        #print "else",(r1,r2,x1,y1,x2,y2)
      index += 1
  xs = []
  ys = []
  for x in line_xs:
    for y in line_ys:
      xs.append(x)
      ys.append(y)
    for m,b in zip(ms,bs):
      xs.append(x)
      ys.append(m*x+b)
  for y in line_ys:
    for m,b in zip(ms,bs):
      xs.append((y-b)/m)
      ys.append(y)
  #print 'x_int:', xs[-1], 'y_int:', ys[-1]
  return [numpy.sum(xs)/len(xs), numpy.sum(ys)/len(ys)]


def localize_spark(times, signals, dt=2.5e-6, live=False):
  mic_coordinates = numpy.array(zip([5, -5, -5, 5], [6, 6, -6, -6]))
  radius = 14.22  # cm
  thickness = 1.37  # cm
  v_s = 3.5e5  # cm/s
  v_p = 3.5e5  # cm/s
  spark_location = numpy.zeros((2))
  damped_signals = numpy.zeros((numpy.shape(signals)[0], numpy.shape(signals)[1]))
  for index,signal in enumerate(signals):
    damped_signals[index] = [signals[index,x]*(math.exp(1-times[x]/2.0e-5)) for x in range(numpy.shape(signals)[1])]
  ordered_mics = order_by_time(times, damped_signals, max_width=20)
  octant = octant_trilateration(damped_signals, ordered_mics, mic_coordinates)
  quadrant = int(octant / 2)
  settling_time = 0.0
  if (live):
    settling_time = dt / 2.0
  return pgl.signal.accumulated_correlation(damped_signals, dt, mic_coordinates,
                                      radius, thickness, v_s, v_p,
                                      grid_size=100, quadrant=quadrant,
                                      settling_time=settling_time)


def OLD_localize_spark_pp(times, signals, v_s, v_p, grid_size, live):
  dt = times[1] - times[0]
  mic_coordinates = numpy.array(zip([5, -5, -5, 5], [6, 6, -6, -6]))
  radius = 14.22  # cm
  thickness = 1.37  # cm
  ordered_mics = order_by_time(times, signals, live)
  """
  damped_signals = numpy.zeros((numpy.shape(signals)[0], numpy.shape(signals)[1]))
  blank_offset = int(22e-6 / dt)
  blank = numpy.zeros(blank_offset)
  for index,signal in enumerate(signals):
      damped_signals[index] = \
        [signals[index,x]*(math.exp(1-times[x]/2.0e-5)) \
         for x in range(numpy.shape(signals)[1])]

      damped_signals[index,:blank_offset] = blank
  signals = damped_signals
  """
  mics = sorted(ordered_mics, key=lambda mic: mic.id())
  """
  for index,signal in enumerate(signals):
    blank_offset = int((mics[index].value()-1.5e-5) / dt)
    blank = numpy.zeros(blank_offset)
    signals[index,:blank_offset] = blank
  """
  #signals = signals_to_cwts(times, signals)
  octant = octant_trilateration(signals, ordered_mics, mic_coordinates)
  #octant,signals = wavelet_conditioning(times, signals, mic_coordinates)
  return pgl.signal.accumulated_correlation(
                            signals, dt,
                            mic_coordinates,
                            radius, thickness,
                            v_s, v_p,
                            grid_size=grid_size,
                            octant=octant)

def condition_signals(times, signals, window_width=50e-6):
  integrations = numpy.cumsum(numpy.abs(signals), axis=1)
  #plot.plot_signals(times*1e6, integrations)

  noise_floor_lines = []
  for integration in integrations:
    coefficients = pgl.curve.fit_polynomial(times[:10]*1e6, integration[:10], order=1)
    line = pgl.curve.polynomial(coefficients, times*1e6)
    noise_floor_lines.append(line)
    #plt.plot(times*1e6, line)
  noise_floor_lines = numpy.array(noise_floor_lines)

  int_polys = []
  for integration in integrations:
    coefficients = pgl.curve.fit_polynomial(times[10:]*1e6, integration[10:], order=4)
    poly = pgl.curve.polynomial(coefficients, times*1e6)
    int_polys.append(poly)
    #plt.plot(times*1e6, poly)
  int_polys = numpy.array(int_polys)

  #plot.plot_signals(times*1e6, (int_polys-noise_floor_lines))
  signs = numpy.sign(int_polys-noise_floor_lines)
  signs[signs==0] = -1
  #print numpy.diff(signs)
  zero_crossings = numpy.where(numpy.diff(signs))
  zero_coords = numpy.unravel_index(zero_crossings, signals[0].shape)[0].transpose()
  #print zero_coords
  last_zero_indicies = zero_coords[
    numpy.append(numpy.where(numpy.diff(zero_coords[:,0]))[0],[-1])].transpose()
  """
  print('Last Zero Indicies:', last_zero_indicies)
  zero_times = times[last_zero_indicies[1]]
  print('shape:',zero_times.shape)
  zero_values = signals[last_zero_indicies[0],last_zero_indicies[1]]
  #print 'shape:',zero_values.shape
  plt.plot(zero_times*1e6, zero_values, '*')
  """
  dt = times[1] - times[0]
  window_sample_count = int(round(window_width/dt))
  prebuffer_sample_count = int(round(10e-6/dt))
  #print 'samples:', window_sample_count
  conditioned_signals = numpy.zeros(signals.shape)
  min_index = signals.shape[1]
  max_index = 0
  for index,signal in enumerate(signals):
    start_index = last_zero_indicies[1,index] - prebuffer_sample_count
    if start_index < min_index:
      min_index = start_index
    end_index = start_index + window_sample_count
    if end_index > max_index:
      max_index = end_index
    conditioned_signals[index,start_index:end_index]\
      = numpy.copy(signal[start_index:end_index])
  return (times[:max_index-min_index], conditioned_signals[:,min_index:max_index])

def signals_to_cwts(times, signals):
  freqs, fmags, fphases = pgl.signal.spectra(times, signals)
  f_peaks = freqs[numpy.argmax(fmags[:,1:], axis=1)]
  #print f_peaks
  dt = times[1] - times[0]
  fc = pgl.signal.ricker_center_freq(dt)
  target_scales = fc / f_peaks
  cwts = numpy.array(
    [scipy.signal.cwt(signals[i], scipy.signal.ricker, [target_scales[i],])[0]
      for i in range(len(signals))])
  return cwts

def wavelet_conditioning(times, signals, mic_coordinates):
  max_width = 20
  M,N = numpy.shape(signals)
  widths = numpy.linspace(1, max_width, num=N)
  mic_tree = pgl.tree.BinarySearchTree()
  dt = times[1] - times[0]
  #cwts = []
  signals = signals_to_cwts(times, signals)
  #for index,signal in enumerate(signals):
  for signal in signals:
    #cwtmatr = scipy.signal.cwt(signals[index], scipy.signal.ricker, widths)
    #indicies = numpy.unravel_index(cwtmatr.argmax(), cwtmatr.shape)
    #cwts.append(cwtmatr[indicies[0]])
    index = signal.argmax()
    #node = pgl.tree.BinaryTreeNode(index, times[indicies[1]])
    node = pgl.tree.BinaryTreeNode(index, times[index])
    mic_tree.insert(node)
  ordered_mics = mic_tree.walk_in_order()
  mics = sorted(ordered_mics, key=lambda mic: mic.id())
  for index,signal in enumerate(signals):
    blank_offset = int((mics[index].value()-1.5e-5) / dt)
    blank = numpy.zeros(blank_offset)
    signals[index,:blank_offset] = blank
  octant = octant_trilateration(signals, ordered_mics, mic_coordinates)
  return (octant, signals)

def localize_spark_pp(times, signals, thetas, grid_size, live, absval=True):
  dt = times[1] - times[0]
  mic_coordinates = numpy.array(zip([5, -5, -5, 5], [6, 6, -6, -6]))
  radius = 14.22  # cm
  thickness = 1.37  # cm
  """
  normalized_signals = numpy.apply_along_axis(
    lambda x: x/numpy.linalg.norm(x), 1, signals)
  ordered_mics = order_by_time(times, normalized_signals, live)
  mics = sorted(ordered_mics, key=lambda mic: mic.id())
  octant = octant_trilateration(ordered_mics)
  #print 'Octant:', octant
  """
  #octant = octant_from_net(signals)
  octant = -11
  if absval:
    signals = numpy.abs(signals)
  return pgl.signal.accumulated_correlation(
                            signals, dt,
                            mic_coordinates,
                            radius, thickness,
                            thetas,
                            grid_size=grid_size,
                            octant=octant)

def localize_spark_sym6(times, signals, thetas, grid_size, live):
  dt = times[1] - times[0]
  mic_coordinates = numpy.array(zip([7.8, 3.9, -3.9, -7.8, -3.9, 3.9],
                                    [0, 6.75, 6.75, 0, -6.75, -6.75]))
  radius = 14.22  # cm
  thickness = 1.37  # cm
  octant = -11
  return pgl.signal.accumulated_correlation(
                            signals, dt,
                            mic_coordinates,
                            radius, thickness,
                            thetas,
                            grid_size=grid_size,
                            octant=octant)

def localize_spark_pp_78(times, signals, thetas, grid_size, live):
  dt = times[1] - times[0]
  mic_coordinates = numpy.array(zip([7.53, -2.02, -7.53, 2.02], [2.02, 7.53, -2.02, -7.53]))
  radius = 14.22  # cm
  thickness = 1.37  # cm
  normalized_signals = numpy.apply_along_axis(
    lambda x: x/numpy.linalg.norm(x), 1, signals)
  """"""
  ordered_mics = order_by_time(times, normalized_signals, live)
  mics = sorted(ordered_mics, key=lambda mic: mic.id())
  octant = octant_trilateration(ordered_mics)
  #print 'Octant:', octant
  """"""
  #octant = octant_from_net(signals)
  return pgl.signal.accumulated_correlation(
                            normalized_signals, dt,
                            mic_coordinates,
                            radius, thickness,
                            thetas,
                            grid_size=grid_size,
                            octant=octant)

def localize_spark_pp_oct(times, signals, thetas, grid_size, octant):
  dt = times[1] - times[0]
  mic_coordinates = numpy.array(zip([5, -5, -5, 5], [6, 6, -6, -6]))
  radius = 14.22  # cm
  thickness = 1.37  # cm
  normalized_signals = numpy.apply_along_axis(
    lambda x: x/numpy.linalg.norm(x), 1, signals)
  return pgl.signal.accumulated_correlation(
                            normalized_signals, dt,
                            mic_coordinates,
                            radius, thickness,
                            thetas,
                            grid_size=grid_size,
                            octant=octant)

def localize_spark_sin(times, signals, dt=2.5e-6, live=False):
  mic_coordinates = numpy.array(zip([5, -5, -5, 5], [6, 6, -6, -6]))
  radius = 14.22  # cm
  thickness = 1.37  # cm
  v_s = 3.5e5  # cm/s
  v_p = 3.5e5  # cm/s
  settling_time = 0.0
  if (live):
    settling_time = dt / 2.0
  return pgl.signal.accumulated_correlation(signals, dt, mic_coordinates,
                                      radius, thickness, v_s, v_p,
                                      grid_size=100,
                                      settling_time=settling_time)
