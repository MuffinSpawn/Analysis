import numpy as np  # NumPy (multidimensional arrays, linear algebra, ...)

def load_csv_data(directory, filename):
  f = open("".join((directory, filename)), "r")
  record_count = int(1)
  line_count = int(0)
  first_x = -1.
  last_x = -1.
  for line in f:
    if line.find("%") == -1:
      line_count += 1
      values = line.split(",")
      x = float(values[0])
      if (first_x < 0):
          first_x = x
      if (x < last_x):
          record_count += 1
      last_x = x

  samples_per_record = int(line_count / record_count)
  print("".join(("Loading ", str(record_count) , " Records...")))
  print("".join(("Samples per Record: ", str(samples_per_record))))

  xs = np.zeros((record_count, samples_per_record))
  ys = np.zeros((record_count, samples_per_record))


  f.seek(0)
  sample_index = int(0)
  record_index = int(0)
  for line in f:
    if line.find("%") == -1:
      values = line.split(",")
      xs[record_index, sample_index] = float(values[0])
      ys[record_index, sample_index] = float(values[1])
      sample_index += 1
      if (sample_index % samples_per_record) == 0:
        sample_index = 0
        record_index += 1
  f.close()
  if (xs[0][1] >= 1.0):
    xs = xs * 1.0e-6

  return (xs, ys)

def load_data(directory, filename, dt=1):
  xs = np.empty(0)
  ys = np.empty(0)
  if ".csv" in filename:
    (xs, ys) = load_csv_data(directory, filename)
    xs = xs[0]
  elif ".npy" in filename:
    ys = np.load("".join((directory, filename)))
    num = len(ys[0,0])
    xs = np.linspace(0, dt*(num-1), num)
  return (xs, ys)
