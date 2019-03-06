import math
import numpy as np  # NumPy (multidimensional arrays, linear algebra, ...)
import numpy.core.records as rec
import re

def load_binary_data(filename, sampling_rate, samples_per_record, sample_size,
                      offset, duration):
  fd = open(filename, 'rb')

  data = load_binary_data_from_fd(fd, sampling_rate, samples_per_record,
                                  sample_size, offset, duration)

  fd.close()

  return data


def load_binary_data_from_fd(fd, sampling_rate, samples_per_record, sample_size,
                      offset, duration):
  # calculate what blocks we need from the file and
  # where in those blocks the data lie
  number_of_channels = int(7)
  time_interval = 1.0 / float(sampling_rate)
  time_per_record = float(samples_per_record) / float(sampling_rate) # 0.1024

  offset_in_samples = int(offset * sampling_rate)
  offset_in_blocks = offset_in_samples / samples_per_record
  sample_offset = offset_in_samples - offset_in_blocks * samples_per_record
  samples_in_first_block_records = samples_per_record - sample_offset

  duration_in_samples = int(duration * sampling_rate)
  duration_in_bytes = duration_in_samples * sample_size
  duration_in_blocks = duration_in_samples / samples_per_record
  samples_in_last_block_records = (duration_in_samples + sample_offset) \
                        % samples_per_record

  extra_samples = duration_in_samples % samples_per_record
  if (extra_samples > 0):
    duration_in_blocks += 1
  if (extra_samples+sample_offset > samples_per_record):
    duration_in_blocks += 1

  record_size = samples_per_record * sample_size
  block_size = record_size * number_of_channels
  data = np.zeros((number_of_channels, duration_in_samples),
                  dtype=np.float64)

  ### load the raw data ###
  #fd = open(filename, 'rb')

  # Skip to offset
  offset_in_bytes = offset_in_blocks * samples_per_record \
                  * number_of_channels * sample_size
  fd.seek(offset_in_bytes)

  # read in the blocks containing the samples we want
  data_string = fd.read(duration_in_blocks*samples_per_record*sample_size*number_of_channels)
  if (len(data_string) == 0):
    return np.zeros((number_of_channels, 0))

  #fd.close()

  # convert byte data to a numpy structure
  block_type = np.dtype([('blocks', np.float64, (number_of_channels, samples_per_record))])
  blocks = rec.fromstring(data_string, dtype=block_type)

  # reshape the data from data blocks to a 2D array of samples for each channel
  append_index = int(0)
  block_count = int(0)
  for block in blocks['blocks']:
    block_count += 1
    record_length = samples_per_record
    if (duration_in_blocks == 1):
      record_length = samples_in_last_block_records - sample_offset
    elif (block_count == 1):
      record_length = samples_in_first_block_records
    elif (block_count == duration_in_blocks):
      record_length = samples_in_last_block_records

    record_index = int(0)
    for record in block:
      if (block_count == 1):
        data[record_index].put(np.arange(append_index,
                                         append_index+record_length,
                                         1, dtype=int),
                                record[sample_offset:sample_offset+record_length])
      else:
        data[record_index].put(np.arange(append_index,
                                         append_index+record_length,
                                         1, dtype=int),
                                record[0:record_length])
      record_index += 1
    append_index += record_length

  return data

"""
waveform,[0],[1],[2],[3],[4],[5],[6]
t0,12/31/1903  18:00:00.000000,12/31/1903  18:00:00.000000,12/31/1903  18:00:00.000000,12/31/1903  18:00:00.000000,12/31/1903  18:00:00.000000,12/31/1903  18:00:00.000000,12/31/1903  18:00:00.000000
delta t,2.000000E-6,2.000000E-6,2.000000E-6,2.000000E-6,2.000000E-6,2.000000E-6,2.000000E-6

time,Y[0],Y[1],Y[2],Y[3],Y[4],Y[5],Y[6]
12/31/1903  18:00:00.000000,-6.650122E-3,-3.309753E-3,-6.814400E-3,6.353830E-3,-5.486458E-3,-2.060305E-3,1.676922E-2
...
"""
def load_csv_data(directory, filename):
  data_re = re.compile('[0-9]')
  f = open("".join((directory, filename)), "r")

  # read the header and count the number of samples per trace
  trace_count = int(0)
  delta_t = float(0.0)
  start_file_position = int(0)
  samples_per_trace = int(0)
  for line in f:
    values = line.split(",")
    if (trace_count == 0):
      trace_count = len(values)-1
    elif (values[0] == "delta t"):
      delta_t = float(values[1])
    elif (values[0] == "time"):
      start_file_position = f.tell()
    elif (data_re.match(line) != None):
      samples_per_trace +=1

  # print "".join(("Loading ", str(trace_count) , " Traces..."))
  # print "".join(("Samples per Trace: ", str(samples_per_trace)))

  xs = np.zeros((trace_count, samples_per_trace))
  ys = np.zeros((trace_count, samples_per_trace))

  line_index = int(0)
  trace_index = int(0)
  f.seek(0)
  for line in f:
    if (line_index >= 5):
      sample_index = line_index - 5
      # print "".join(("line: ", line))
      values = line.split(",")
      # handle date field
      # print "".join(("timestamp: ", str(values[0])))
      seconds = values[0].split(":")[2];
      for trace_index in range(len(values)-1):
        xs[trace_index, sample_index] = float(seconds)
      trace_index = int(0)
      for value in values[1:]:
        ys[trace_index, sample_index] = float(value)
        trace_index += 1
    line_index += 1
  f.close()

  return (xs, ys)

def load_numpy_data(directory, filename):
  npz_data = np.load("".join((directory, filename)))
  tags = np.sort(npz_data.files)
  prefix = tags[0].split('_')[0]
  indicies = np.zeros(len(tags), dtype="int")
  for i in range(len(tags)):
    indicies[i] = int(tags[i].split('_')[1])
  indicies = np.sort(indicies)
  for i in range(len(tags)):
    tags[i] = ''.join((prefix, '_', str(indicies[i])))
  xs = np.array([npz_data[tag][0] for tag in tags])
  ys = np.array([npz_data[tag][1] for tag in tags])
  return (xs, ys)

def save_numpy_data(directory, filename, times, signals, t0=-1, t1=-1):
  start_index = 0
  dt = times[1] - times[0]
  if t0 > 0:
    start_index =  math.ceil((t0-times[0]) / dt)
  signals = signals[:,start_index:]
  if t1 >= 0:
    end_index = math.floor((t1-t0) / dt)
    signals = signals[:,:end_index]
  """
  np.savez(''.join((directory, filename)),
           np.vstack((times, signals[0])),
           np.vstack((times, signals[1])),
           np.vstack((times, signals[2])),
           np.vstack((times, signals[3])),
           np.vstack((times, signals[4])),
           np.vstack((times, signals[5])),
           np.vstack((times, signals[6])))
  """
  np.save(''.join((directory, filename)), [signals])

def load_data(directory, filename, t0=0, t1=-1, channels=[], adjust=False):
  xs = np.empty(0)
  ys = np.empty(0)
  if ".csv" in filename:
    (xs, ys) = load_csv_data(directory, filename)
  elif ".npz" in filename:
    (xs, ys) = load_numpy_data(directory, filename)
  else:
    return None
  dt = xs[0,1] - xs[0,0]
  if (t0 >= 0):
    start_index = int(t0 / dt)
  else:
    start_index = 0
  if (t1 > 0):
    end_index = int(t1 / dt)
  else:
    end_index = -1
  if len(channels) == 0:
    channels = range(len(ys))
  times = xs[0, start_index:end_index] - xs[0,start_index]
  signal_list = []
  offset = 0
  for ch in channels:
    """
    if adjust and ch == 2:
      offset = 1
    else:
      offset = 0
    """
    """
    if adjust:
      offset = int(ch) / 2
    """
    if adjust and ch > 3:
      offset = 1
    signal_list.append(ys[ch,start_index+offset:end_index+offset])
  signals = np.array(signal_list)
  return (times, signals)
