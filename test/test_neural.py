# -*- coding: utf-8 -*-
"""
Created on Mon May 16 08:45:59 2016

@author: lane
"""

import copy
import math
import subprocess as proc
import sys
import platform
import numpy as np  # NumPy (multidimensional arrays, linear algebra, ...)
import numpy.linalg as linalg
import numpy.random as rand
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.collections as mcollections
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt  # Matplotlib's pyplot: MATLAB-like syntax
import scipy.io as io
import scipy.optimize as op
import scipy.signal as sig
import scipy.special as special
import bisect as bi
import time
import pgl.neural as neural
import pgl.progress as prog
import pp
from functools import partial

def reset_plot_params():
    mpl.rcParams['ytick.labelsize'] = 22
    mpl.rcParams['xtick.labelsize'] = 22
    mpl.rcParams['axes.labelsize'] = 26
    mpl.rcParams['font.size'] = 26
    mpl.rcParams['mathtext.default'] = 'regular'
    mpl.rcParams['figure.subplot.left'] = 0.02
    mpl.rcParams['figure.subplot.right'] = 0.98
    mpl.rcParams['figure.subplot.top'] = 0.9
    mpl.rcParams['figure.subplot.bottom'] = 0.1
    mpl.rcParams['figure.subplot.wspace'] = 0.2
    mpl.rcParams['figure.subplot.hspace'] = 0.2
reset_plot_params()

""" Ported from Andrew Ng's Introduction to Machine Learning Assignment 4
    function of the same name.
"""
def debugInitializeWeights(out_size, in_size):
  weights = np.empty((out_size, in_size+1))
  return np.reshape(np.sin(np.arange(weights.size)), weights.shape) / 10;

""" Ported from Andrew Ng's Introduction to Machine Learning Assignment 4
    function of the same name.
"""
def computeNumericalGradient(obj_func, Theta):
  JacApx = np.empty(len(Theta), dtype=np.ndarray)
  perturb = np.empty(len(Theta), dtype=np.ndarray)
  eps = 1e-4
  for l,theta_l in enumerate(Theta):
    JacApx[l] = np.zeros(theta_l.shape)
    perturb[l] = np.zeros(theta_l.shape)
  for l,theta_l in enumerate(Theta):
    for i,theta_i in enumerate(theta_l):
      for j,theta_ij in enumerate(theta_i):
        perturb[l][i,j] = eps
        Theta_plus = np.empty(len(Theta), dtype=np.ndarray)
        Theta_minus = np.empty(len(Theta), dtype=np.ndarray)
        for ll,theta in enumerate(Theta):
          Theta_plus[ll] = theta + perturb[ll]
          Theta_minus[ll] = theta - perturb[ll]
        J_plus,_ = obj_func(Theta_plus)
        J_minus,_ = obj_func(Theta_minus)
        JacApx[l][i,j] = (J_plus - J_minus)\
                       / (2*eps)
        perturb[l][i,j] = 0
  return JacApx

def check_gradients():
  input_layer_size = 3
  hidden_layer_size = 5
  num_labels = 3
  m = 5
  
  # We generate some 'random' test data
  Theta1 = debugInitializeWeights(hidden_layer_size, input_layer_size)
  Theta2 = debugInitializeWeights(num_labels, hidden_layer_size)
  Theta = np.array((Theta1, Theta2))
  # Reusing debugInitializeWeights to generate X
  X  = debugInitializeWeights(m, input_layer_size - 1)
  y  = np.arange(0, m) % num_labels
  Y = np.zeros((m, num_labels))
  Y[np.arange(5),y]=1

  # Unroll parameters
  #nn_params = [Theta1(:) ; Theta2(:)];
  
  # Short hand for cost function
  net = neural.Net([input_layer_size, hidden_layer_size, num_labels])
  cost,grad = net.objective_function(Theta, X, Y)
  flatgrad = np.hstack([x.flatten() for x in grad])[:,np.newaxis]
  
  numgrad = computeNumericalGradient(
    lambda theta: net.objective_function(theta, X, Y), Theta)
  flatnumgrad = np.hstack([x.flatten() for x in numgrad])[:,np.newaxis]
  
  # Visually examine the two gradient computations.  The two columns
  # you get should be very similar. 
  #print "Numerical Gradient:", flatnumgrad
  #print "Backprop Gradient:", flatgrad
  print np.hstack([flatnumgrad, flatgrad])
  print '\n'.join((
    'The above two columns you get should be very similar.',
    '(Left-Your Numerical Gradient, Right-Analytical Gradient)\n'))
  
  # Evaluate the norm of the difference between two solutions.  
  # If you have a correct implementation, and assuming you used EPSILON = 0.0001 
  # in computeNumericalGradient.m, then diff below should be less than 1e-9
  diff = linalg.norm(flatnumgrad-flatgrad)/linalg.norm(flatnumgrad+flatgrad);
  
  print '\n'.join((
    'If your backpropagation implementation is correct, then',
    'the relative difference will be small (less than 1e-9).'
    '\nRelative Difference:')), diff


if __name__ == '__main__':
  """
  X = np.array([[0, 1]]).transpose()
  print 'Inputs:\n', X, '\n'

  Y = np.array([[1, 0]]).transpose()
  net = Net([1, 1, 1])
  print net
  #flat_weights = net.train(X, Y)
  flat_weights = net.train_iter(X, Y, 10)
  weights = net.rollup_weights(flat_weights)
  print 'Final Weights:', weights
  print net.objective_function(weights, X[:1], Y[0:1])
  for x in X:
    print net.forward_propagate(x) >= 0.5

  cost_history = net._cost_history
  print cost_history
  plt.plot(cost_history)
  """
  """"""
  X = np.array([[0,0],[1,0],[0,1],[1,1]])
  print 'Inputs:\n', X, '\n'

  # Train OR
  print '### OR ###'
  Y = np.array([[0,1,1,1]]).transpose()
  net = neural.Net([2, 1])
  #net.train_iter(X, Y, 100, lam=0)
  net.train(X, Y, lam=0)
  for x in X:
    print net.forward_propagate(x) >= 0.5

  # Train AND
  print '\n### AND ###'
  Y = np.array([[0,0,0,1]]).transpose()
  net = neural.Net([2, 1])
  #net.train_iter(X, Y, 5)
  net.train(X, Y, lam=0)
  for x in X:
    print net.forward_propagate(x) >= 0.5

  # Train NOR
  print '\n### NOR ###'
  Y = np.array([[1,0,0,0]]).transpose()
  net = neural.Net([2, 1])
  #print net
  #net.train_iter(X, Y, 10)
  net.train(X, Y, lam=0)
  #print net
  for x in X:
    print net.forward_propagate(x) >= 0.5

  # Train XOR
  print '\n### XOR ###'
  Y = np.array([[0,1,1,0]]).transpose()
  net = neural.Net([2, 2, 1])
  #net.train_iter(X, Y, 1000)
  net.train(X, Y, lam=0)
  #net.train(X, Y)
  for x in X:
    print net.forward_propagate(x) >= 0.5
  """"""

  """
  weights_file = ''.join(('/home/lane/Development/Coursera/',
                          'IntroToMachineLearning/machine-learning-ex4/ex4/',
                          'ex4weights.mat'))
  weights_data = io.loadmat(weights_file)
  Theta1 = weights_data['Theta1'] 
  Theta2 = weights_data['Theta2']
  ex4net = neural.Net((400, 25, 10))
  #ex4net.set_weights(np.array([Theta1, Theta2]))

  inputs_file = ''.join(('/home/lane/Development/Coursera/',
                         'IntroToMachineLearning/machine-learning-ex4/ex4/',
                         'ex4data1.mat'))
  inputs_data = io.loadmat(inputs_file)
  X = inputs_data['X']
  y = inputs_data['y']

  Y = np.zeros((len(y), 10))
  for i,output in enumerate(y):
    Y[i,output-1] = 1

  print 'Cost Function Without Regularization:',\
        ex4net.objective_function(np.array([Theta1, Theta2]), X, Y, lam=0)[0]

  print 'Cost Function With Regularization:',\
        ex4net.objective_function(np.array([Theta1, Theta2]), X, Y, lam=1)[0]

  check_gradients()
  """