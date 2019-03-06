#!/usr/bin/python

import sys
import copy
import json;
import matplotlib.pyplot as plt
import numpy as np  # NumPy (multidimensional arrays, linear algebra, ...)
import numpy.linalg as linalg
import numpy.random as rand
import scipy as sp  # SciPy (signal and image processing library)
import scipy.optimize as op
import scipy.io as io

import os.path
import sys
import re
import math
from bisect import bisect_left, bisect_right

class SizeError(Exception):
  def __init__(self, name, expected, found):
    self._message = ' '.join(('Expected size', str(expected), 'for', str(name),
                             'but found', str(found)))
  def __str__(self):
    return self._message

def sigmoid(z):
  return 1.0 / (1.0 + np.exp(-z))

class Layer:
  def __init__(self, size, index):
    self._size = size
    self._activations = np.zeros(size)
    self._index = index
    self._weights = None

  def activations(self):
    return self._activations.copy()

  def set_activations(self, activations):
    self._activations = activations.copy()

  def weights(self):
    return self._weights.copy()

  def set_weights(self, weights):
    self._weights = weights.copy()

  def propagate(self, next_layer):
    a = np.vstack([1, self._activations])
    """
    print('a.shape:', a.shape)
    print 'type(a):', type(a)
    print a
    print '_weights.shape:', self._weights.shape
    print 'type(_weights):', type(self._weights)
    print self._weights
    """
    z = np.dot(self._weights, a)
    #print 'z:', z
    next_layer.set_activations(sigmoid(z))


class Net:
  def __init__(self, layer_sizes):
    self._layers = []
    self._max_size = np.max(layer_sizes)
    for l,layer_size in enumerate(layer_sizes):
      self._layers.append(Layer(layer_size, l))
      if l > 0:
        # initialize the weights for the previous layer
        last_layer = self._layers[l-1]
        last_layer.set_weights(np.zeros((layer_size, last_layer._size+1)))
    self._cost_history = []

  def activations(self):
    activations = []
    for l,layer in enumerate(self._layers):
      activations.append(layer.activations())
    return np.array(activations)

  def weights(self):
    weights = []
    for l,layer in enumerate(self._layers[:-1]):
      weights.append(layer.weights())
    return np.array(weights)

  def set_weights(self, weights):
    if len(weights) != len(self._layers)-1:
      raise SizeError('len(weights)', len(self._layers)-1, len(weights))
    for l,layer in enumerate(self._layers[:-1]):
      if weights[l].shape != (self._layers[l+1]._size, layer._size+1):
        raise SizeError(
          ''.join(('weights[', str(l), '].shape')),
          (self._layers[l+1]._size, layer._size+1),
          weights[l].shape)
      layer.set_weights(weights[l])

  def forward_propagate(self, x, weights=None):
    if weights != None:
      #print 'Forward Prop. Weights:', weights
      #print 'Weights Before:', self.weights()
      self.set_weights(weights)
      #print 'Weights After:', self.weights()

    if len(x) != self._layers[0]._size:
      raise SizeError('x', self._layers[0]._size, len(x))
    self._layers[0].set_activations(x[:,np.newaxis])
    for l,layer in enumerate(self._layers[:-1]):
      layer.propagate(self._layers[l+1])
    return self._layers[-1].activations()

  """ FIXME
  def backpropagate(self, X, Y, weights=None):
    if weights == None:
      for l,layer in enumerate(self._layers[:-1]):
        s_l = layer._size
        s_lplus1 = self._layers[l+1]._size
        layer.set_weights(np.random.random((s_lplus1, s_l+1)))
    else:
      self.set_weights(weights)

    if type(X) != np.ndarray:
      X = np.array(X)
    if len(X.shape) != 2:
      raise SizeError('len(X.shape)', 2, len(X.shape))
    if X.shape[1] != self._layers[0]._size:
      raise SizeError('X.shape[1]', self._layers[0]._size, X.shape[1])

    if type(Y) != np.ndarray:
      Y = np.array(Y)
    if len(Y.shape) != 2:
      raise SizeError('len(Y.shape)', 2, len(Y.shape))
    if Y.shape[1] != self._layers[-1]._size:
      raise SizeError('Y.shape[1]', self._layers[-1]._size, Y.shape[1])

    L = len(self._layers)
    Deltas = []
    for l,layer in enumerate(self._layers[:-1]):
      Deltas.append(np.zeros((self._layers[l+1]._size, layer._size+1)))
    for x,y in zip(X,Y):
      a_L = self.forward_propagate(x, weights=weights)
      #print 'a_L:', a_L
      #print 'y:', y
      delta = (a_L - y)[:,np.newaxis]
      #print 'delta_L', delta
      #print 'delta_L Shape', delta.shape
      #deltas.append(delta)
      #a = np.hstack((1, a_L))[:,np.newaxis]
      for _,layer in enumerate(self._layers[-2::-1]):
        weights_l = layer.weights()
        a = np.hstack((1, layer.activations()))[:,np.newaxis]
        #print 'a:', a
        Deltas[layer._index] += np.dot(delta, a.transpose())
        if (layer._index > 0):
          """"""
          #print 'weights_l^T:', weights_l.transpose()
          #print 'weights_l^T Shape:', weights_l.transpose().shape
          #print 'delta:', delta
          #print 'delta Shape:', delta.shape
          #print 'a:', a
          #print 'weights_l^T.delta:', np.dot(weights_l.transpose(), delta[np.newaxis,:])
          """"""
          aT = a[:,np.newaxis]
          #print 'aT:', aT
          delta = (np.dot(weights_l.transpose(), delta[np.newaxis,:])
                * aT*(1-aT))[1]
          #print 'delta:', delta
          #deltas.append(delta[1:])
    return np.array(Deltas)
  """

  def objective_function(self, weights, X, Y, lam=1):
    L = len(self._layers)
    Delta = np.empty(L-1, dtype=np.ndarray)
    for l,theta_l in enumerate(weights):
      Delta[l] = np.zeros(theta_l.shape)
    activations = []
    J = 0
    m = len(X)
    # Sum over m training examples
    for j in range(m):
      x = X[j]
      y = Y[j]
      """
      A = np.empty(L, dtype=np.ndarray)
      A[0] = x[:,np.newaxis]
      # Loop over all non-output layers and calculate the unit activations
      for l,theta_l in enumerate(weights):
        z_lplus1 = np.dot(theta_l, np.vstack([1, A[l]]))
        A[l+1] = sigmoid(z_lplus1)
      #activations.append(A)
      """
      self.forward_propagate(x, weights)
      A = self.activations()

      # Add the cost contribution of this training example
      #print A
      #print 'y.shape:', y.shape
      #print 'A[-1].shape:', A[-1].shape
      J += -(np.dot(y, np.log(A[-1])) + np.dot((1-y), np.log(1-A[-1])))/m

      # Backpropagate the errors
      a = A[-1]
      delta = a - y[:,np.newaxis]
      for l in range(L-2, -1, -1):
        a = np.vstack([1, A[l]])
        Delta[l] += np.dot(delta, a.transpose())
        if l > 0:
          delta = (np.dot(weights[l].transpose(), delta) * a * (1-a))[1:,:]

    # Add regularization contribution to the cost
    for w in weights:
      J += lam*np.sum(w[:,1:]**2) / (2*m)

    Jacobian = np.empty(len(weights), dtype=np.ndarray)
    Theta = copy.deepcopy(weights)
    for l,D in enumerate(Delta):
      Theta[l][:,0] = 0
      Jacobian[l] = (D + lam * Theta[l]) / m

    return [J, Jacobian]

  def rollup_weights(self, flat_weights):
      index = 0
      weights = []
      for l,layer in enumerate(self._layers[:-1]):
        s_lplus1 = self._layers[l+1]._size
        s_l = layer._size
        length = s_lplus1*(s_l+1)
        weights.append(flat_weights[index:index+length].reshape((s_lplus1, s_l+1)))
        index += length
      return weights

  def train(self, X, Y, lam=1, tol=1e-8, maxiter=1000):
    def J(theta):
      #sys.stdout.write('. ')
      weights = self.rollup_weights(theta)
      [cost, jac] = self.objective_function(np.array(weights), X, Y, lam=lam)
      #print [cost, np.hstack([d.flatten() for d in jac])]
      print(cost[0])
      return [cost[0], np.hstack([d.flatten() for d in jac])]
    initial_theta = np.hstack([w.flatten() for w in self.weights()])
    eps = math.sqrt(6) / math.sqrt(X.shape[1] + Y.shape[1])
    initial_theta = np.random.random(initial_theta.shape)*(2*eps) - eps
    #print 'Initial Theta:', initial_theta
    theta = initial_theta
    self._cost_history = []
    method = 'Newton-CG'
    theta = op.minimize(
              J, theta, method=method, jac=True,
              options={'xtol': tol, 'disp': True, 'maxiter': maxiter}).x
    return self.rollup_weights(theta)

  def train_iter(self, X, Y, iterations, lam=1):
    self._cost_history = []
    #Deltas = self.backpropagate(X, Y)
    weights = np.empty(len(self._layers)-1, dtype=np.ndarray)
    eps = math.sqrt(6) / math.sqrt(X.shape[1] + Y.shape[1])
    for l,layer in enumerate(self._layers[:-1]):
      s_l = layer._size
      s_lplus1 = self._layers[l+1]._size
      weights[l] = np.random.random((s_lplus1, s_l+1))*2*eps - eps
    J, Jacobian = self.objective_function(weights, X, Y, lam=lam)
    for i in range(iterations):
      print(J,Jacobian)
      #Deltas = self.backpropagate(X, Y, weights=weights-1.0*Deltas)
      for l,Jac_l in enumerate(Jacobian):
        weights[l] -= 1.0*Jac_l
      J, Jacobian = self.objective_function(weights, X, Y, lam=lam)
    return np.hstack([w.flatten() for w in weights])

  def __str__(self):
    activations = '\n'.join([str(layer.activations()) for layer in self._layers])
    weights = '\n'.join([str(layer.weights()) for layer in self._layers[:-1]])
    return ''.join(('activations:\n', activations,
                    '\nweights:\n', weights))
