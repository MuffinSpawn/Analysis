#!/usr/bin/python

import math
import numpy as np  # NumPy (multidimensional arrays, linear algebra, ...)
import matplotlib.pyplot as plt  # Matplotlib's pyplot: MATLAB-like syntax
import pgl.cluster as cluster
import pgl.curve as curve
import pgl.signal as signal
import scipy.special as sp


def plot_signals(times, signals, tlim=-1, ylim=-1, norm=False, axes=None):
  if axes == None:
    axes = plt.gca()
  if (type(tlim) == tuple):
    axes.set_xlim(tlim)
  elif tlim >= 0:
    axes.set_xlim((0, tlim))
  if type(ylim) == tuple:
    axes.set_ylim(ylim)
  elif ylim > 0:
    axes.set_ylim((-ylim, ylim))
  for signal in signals:
    if norm:
      axes.plot(times, signal / np.max(signal))
    else:
      axes.plot(times, signal, 'b-')

def plot_spectra(times, signals, flim=-1, ylim=-1, norm=False, prnt=False):
  dt = times[1] - times[0]
  df = 1.0 / dt / len(times)
  if (prnt):
    print(''.join(("df: ", str(df))))
  if (type(flim) == tuple):
    plt.xlim(flim)
  elif flim > 0:
    plt.xlim((0, flim))
  if (type(ylim) == tuple):
    plt.ylim(ylim)
  elif ylim > 0:
    plt.ylim((0, ylim))
  (frequencies, magnitudes, phases) = signal.spectra(times, signals)
  for magnitude in magnitudes:
    if norm:
      plt.plot(frequencies, magnitude / np.max(magnitude))
    else:
      plt.plot(frequencies, magnitude)
    """
    if (prnt):
      print ''.join(("Freq. Peaks: ", str(peaks(frequencies[:int(250e3/df)], magnitude[:int(250e3/df)]))))
    """

"""
Plot the average spectra of the given signals along with Gaussian fits to the peaks.
"""
def plot_spectral_average(times, signals, flim=250e3, ylim=(-1,1), prnt=True):
    dt = times[1] - times[0]
    df = 1.0 / dt / len(times)
    if (prnt):
        print(''.join(("df: ", str(df))))
    if (type(flim) == tuple):
        plt.xlim(flim)
    else:
        plt.xlim((0, flim))
    (frequencies, magnitudes, phases) = signal.spectra(times, signals)
    # spectrum_magnitudes = spectrum_magnitudes / spectrum_magnitudes.max()
    shortened_length = round(len(frequencies)/20.0)
    frequencies = frequencies[1:shortened_length]
    spectrum_magnitudes = spectrum_magnitudes[1:shortened_length]
    spectra_phases[1:,:shortened_length]

    (cluster_assignments, centroids, objective) \
      = cluster.kmeans_peak_cluster(frequencies, spectrum_magnitudes, K=4)
    plt.scatter(frequencies, spectrum_magnitudes, s=10*(cluster_assignments+1),
                c=10*(cluster_assignments+1), alpha=0.5)
    #(cluster_assignments, centroids, objective) = kmeans_peak_cluster(frequencies, spectrum_magnitudes, K=2)
    #plt.scatter(frequencies, spectrum_magnitudes, s=10*(cluster_assignments+1), c=10*(cluster_assignments+1), alpha=0.5)

    #plt.plot(frequencies, spectrum_magnitudes)
    gaussian_fits = cluster.spectral_cluster(
                      frequencies,
                      spectrum_magnitudes / spectrum_magnitudes.max(),
                      krange=(4,8))
    for gaussian_fit in gaussian_fits:
        A = gaussian_fit[0]
        B = gaussian_fit[1]
        mu = gaussian_fit[2]
        sigma = gaussian_fit[3]
        plt.plot(frequencies, map(lambda f: curve.gaussian(f, A, B, mu, sigma)*spectrum_magnitudes.max(), frequencies))
    means = gaussian_fits[:,2]
    """
    print ''.join(('frequencies: ', str(frequencies)))
    print ''.join(('means: ', str(means)))
    """
    peak_magnitudes = map(lambda mu: spectrum_magnitudes[np.where(frequencies==mu)[0][0]], means)
    peaks = np.array(sorted(zip(means, peak_magnitudes), key=lambda tup: tup[1], reverse=True))
    peak_t_offsets = np.empty((len(signals), len(peaks)))
    for index,spectrum_phases in enumerate(spectra_phases):
        peak_phases = map(lambda f: spectrum_phases[np.where(frequencies==f)[0][0]], peaks[:,0])
        peak_t_offsets[index] = np.array(map(lambda delta,f: delta / (2*math.pi*f), peak_phases, peaks[:,0]))
    if (prnt):
        print(''.join(("Peak Frequencies: ", str(peaks))))
        print(''.join(("Peak Phases: ", str(peak_t_offsets))))

def plot_formula(formula, a, b, num=50):
  x = np.linspace(a, b, num)
  y = eval(formula)
  plt.plot(x, y)
