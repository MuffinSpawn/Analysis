import numpy as np  # NumPy (multidimensional arrays, linear algebra, ...)
import numpy.core.records as rec
import re

def load_csv_data(directory, filename):
  data_re = re.compile('^[^,]*,[^,]*,[^,]*,')
  f = open("".join((directory, filename)), "r")

  # read the header
  trace_count = int(0)
  samples_per_trace = int(0)

  line_index = int(0)
  trace_index = int(0)
  sample_index = int(0)
  for line in f:
    # print "".join(("line: ", line))
    values = line.split(",")
    if (line_index == 0):
      trace_count = len(values) - 5
      samples_per_trace = int(float(values[1]))
      xs = np.zeros((trace_count, samples_per_trace))
      ys = np.zeros((trace_count, samples_per_trace))
    # handle date field
    # print "".join(("timestamp: ", str(values[0])))
    seconds = values[3]
    for trace_index in range(trace_count):
      xs[trace_index, sample_index] = float(seconds)
    trace_index = int(0)
    for value in values[4:len(values)-1]:
      ys[trace_index, sample_index] = float(value)
      trace_index += 1
    line_index += 1
    sample_index += 1
  f.close()

  return (xs, ys)