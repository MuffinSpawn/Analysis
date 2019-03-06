#!/usr/bin/python

import json;
import matplotlib.pyplot as plt
import numpy as np  # NumPy (multidimensional arrays, linear algebra, ...)
import numpy.linalg as linalg
import numpy.random as rand
import scipy as sp  # SciPy (signal and image processing library)
import scipy.optimize as op

import os.path
import sys
import re
import math
from bisect import bisect_left, bisect_right

def sum_array(data):
  if (np.size(data) == 0):
    return 0
  return data[0] + sum_array(data[1:])

def mean(data):
  return sum_array(data) / data.size

def variance(data, mean):
  return sum_array(map((lambda x: x**2), map((lambda x: x-mean), data))) \
         / (data.size-1)

def stddev(data, mean):
  return math.sqrt(variance(data, mean))

"""
def polynomial(X, a, b):
  if (a.size == 0):
    return 0
  def polynomial_term(x, a_i, b_i, order):
    return a_i * (x-b_i)**order
  return map(lambda x: np.sum(np.fromiter(map((lambda a_i,b_i,i:
      polynomial_term(x, a_i, b_i, i)), a, b, np.arange(a.size)), dtype=np.float)), X)
"""

def polynomial(cs, xs):
  if (cs.size == 0):
    return 0
  def polynomial_term(x, c_i, order):
    return c_i * x**order
  return np.fromiter(map(lambda x: np.sum(np.fromiter(map((lambda c_i,i:
      polynomial_term(x, c_i, i)), cs, np.arange(cs.size)), dtype=np.float)), xs), dtype=np.float)

def polynomials(segments, coefficient_sets, xs):
  poly_segs = np.empty(0)
  for segment,coefficient_set in zip(segments,coefficient_sets):
      poly_segs = np.append(poly_segs, polynomial(coefficient_set, xs[segment[0]:segment[1]]))
  return poly_segs

def gaussian(x, A, B, mu, sigma):
  return A + B/np.sqrt(2*math.pi*sigma**2) \
             * np.exp(-(x-mu)**2/(2*sigma**2))

def gaussian_ssr(gaussian_fit, X, Y):
    A = gaussian_fit[0]
    B = gaussian_fit[1]
    mu = gaussian_fit[2]
    sigma = gaussian_fit[3]
    return np.sum(np.fromiter(map(lambda x,y: (gaussian(x, A, B, mu, sigma)-y)**2, X, Y), dtype=np.float))

def reciprocal(cs, xs):
  if (cs.size == 0):
    return 0
  return np.fromiter(map(lambda x: cs[0] + cs[1] / x, xs), dtype=np.float)

"""
X are parameters and Y are corresponding measurements with estimators
f(x_i) = sum_{j=1...m}(b_j h_j(x_i)), where h_j are m linearly independent
functions. To find the coefficients, b_j, we must solve the linear equation
Y = H B. The least squares solution is the Moore-Penrose Psuedo Inverse of H
times Y: B = (H^T H)^-1 H^T Y = D Y.

In this case h_j(x_i) is a Gaussian. To make the least squares method work we
must first calculate the mean and standard deviation. The mean is found by
treating y_i as a histogram count and then calculating sum(x_i y_i)/sum(y_i).
The standard deviation can be found by solving the Gaussian for it when y_i is
half the max height (when x_i = mean). Since the true mean may not be exactly
where the data mean is, the "adjust" parameter can be used to shift the
calculated mean so as to achieve a better fit.
"""
def fit_gaussian(X, Y, adjust=0):
  # Make an educated guess at the mean
  mu = sum_array(map((lambda x,y: x*y), X, Y)) / sum_array(Y)
  return fit_gaussian_with_mean(X, Y, mu, adjust)

def fit_gaussian_with_mean(X, Y, mu=mean, yerr=1, adjust=0, yoff=True):
  # Determine whether f(mu) should be a peak or trough
  dydx = np.gradient(Y)
  d2ydx2 = np.gradient(dydx)
  concavity = d2ydx2[d2ydx2.size/2]
  # Calculate sigma from average |x| value at half max
  half_max = (np.max(Y) + np.min(Y))/2
  x0 = 0.
  if (concavity < 0.):   # concave downward
    for i in range(X.size)[::-1]:
      if (Y[i] >= half_max):
        x0 = X[i]
        break
    if (x0 == mu):
      for i in range(X.size):
        if (Y[i] >= half_max):
          x0 = 2*mu - X[i]
          break
  else:               # concave upward
    for i in range(X.size)[::-1]:
      if (Y[i] <= half_max):
        x0 = X[i]
        break
    if (x0 == mu):
      for i in range(X.size):
        if (Y[i] <= half_max):
          x0 = 2*mu-X[i]
          break
  x0 += (0.081)*adjust
  # print "".join(("x0: ", str(x0)))
  sigma = (x0-mu)/math.sqrt(math.log(4))
  if sigma == 0:
    sigma = 0.1
  def theta1(x):
    return 1
  def theta2(x):
    return gaussian(x, 0, 1, mu, sigma)
  Ht = np.empty(0)
  if yoff:
    Ht = np.fromiter([map(theta1, X),map(theta2, X)], dtype=np.float)
  else:
    Ht = np.fromiter([map(theta2, X),], dtype=np.float)
  H = np.transpose(Ht)
  if type(yerr) == int:
    yerr = np.ones(len(X))
  V = yerr*yerr*np.identity(len(yerr))
  # print linalg.det(V)
  Vi = linalg.inv(V)
  D = np.dot(linalg.inv(np.dot(Ht,np.dot(Vi,H))),np.dot(Ht,Vi))
  b = np.dot(D, Y)
  parameters = np.empty(0)
  if yoff:
    parameters =  np.array((b[0], b[1], mu, sigma))
  else:
    parameters =  np.array((0, b[0], mu, sigma))
  return parameters

def fit_reciprocal(X, Y, yerr=1):
  def theta1(x):
    return 1.0
  def theta2(x):
    return 1.0/x
  Ht = np.fromiter([map(theta1, X),map(theta2, X)], dtype=np.float)
  H = np.transpose(Ht)
  # D = linalg.pinv(H)
  if type(yerr) == int:
    yerr = np.ones(len(X))
  V = yerr*yerr*np.identity(len(yerr))
  Vi = linalg.inv(V)
  D = np.dot(linalg.inv(np.dot(Ht,np.dot(Vi,H))),np.dot(Ht,Vi))
  b = np.dot(D, Y)
  return np.array((b[0], b[1]))

"""
X are parameters and Y are corresponding measurements with estimators
f(x_i) = sum_{j=1...m}(b_j h_j(x_i)), where h_j are m linearly independent
functions. To find the coefficients, b_j, we must solve the linear equation
Y = H B. The least squares solution is the Moore-Penrose Psuedo Inverse of H
times Y: B = (H^T H)^-1 H^T Y = D Y.

In this case h_j(x_i) = x_i^j.
"""
def fit_polynomial(X, Y, order=2, yerr=1):
  # print("".join(("Shape of X: ", str(np.shape(X)))))
  H = np.empty((X.size, order+1))
  # print("".join(("Initial shape of H: ", str(np.shape(H)))))
  for j in range(order+1):
    H[:,j] = X**j
  # print("".join(("Final shape of H: ", str(np.shape(H)))))
  # print("".join(("H:\n", str(H))))
  # return np.array((0))
  D = np.empty((0,0))
  if type(yerr) == int:
    yerr = np.ones(len(X))
    D = linalg.pinv(H)
  else:
    V = yerr*yerr*np.identity(len(yerr))
    Vi = linalg.inv(V)
    Ht = np.transpose(H)
    D = np.dot(linalg.inv(np.dot(Ht,np.dot(Vi,H))),np.dot(Ht,Vi))
  B = np.dot(D, Y)
  # print "".join(("Shape of B: ", str(np.shape(B))))
  # print B
  return B

def fit_spliced_polynomials(X, Y, order, segments):
  fits = map(lambda segment:
    fit_polynomial(X[segment[0]:segment[1]], Y[segment[0]:segment[1]], order),
    segments)
  return fits

# Perform a linear fit using gradient descent with adaptive learning rate
def fit_line_GD(X, Y):
  m = len(X)
  def J(thetas, xs, ys):
    return np.sum((thetas[0]+thetas[1]*xs-ys)**2)/(2*m)
  def dJ0(thetas, xs,ys):
    return np.sum(thetas[0]+thetas[1]*xs-ys)/m
  def dJ1(thetas, xs,ys):
    return np.sum((thetas[0]+thetas[1]*xs-ys)*xs)/m
  alpha = 1.0e9  # initial learning rate
  max_residual = 1e-6
  theta_old = np.zeros(2)
  theta = np.ones(2)
  residuals = []
  iter_count = int(0)
  old_J = J(theta_old, X, Y)
  while np.any(np.abs(theta - theta_old) > max_residual):
    theta_old = theta
    grad_J = np.fromiter([dJ0(theta_old, X, Y), dJ1(theta_old, X, Y)], dtype=np.float)
    theta = theta_old - alpha*grad_J

    # adapt the learning rate by performing a backtracking line search
    while (alpha > 0.0) and ((old_J - J(theta, X, Y)) <= 0.0):
      alpha = alpha / 2.0
      theta = theta_old - alpha*grad_J
    old_J = J(theta, X, Y)
    #print 'alpha:', alpha

    residuals.append(np.abs(theta-theta_old))
    iter_count += 1
  return theta,residuals

# Perform a linear fit using stochastic gradient descent
# with adaptive learning rate
def fit_line_SGD(X, Y):
  m = len(X)
  def J(thetas, x, y):
    return (thetas[0]+thetas[1]*x-y)**2/2
  def dJ0(thetas, x,y):
    return thetas[0]+thetas[1]*x-y
  def dJ1(thetas, x,y):
    return (thetas[0]+thetas[1]*x-y)*x
  alpha = 1.0e9  # initial learning rate
  max_residual = 1e-6
  theta_old = np.zeros(2)
  theta = np.ones(2)
  residuals = []
  iter_count = int(0)
  old_J = J(theta_old, X[0], Y[0])
  while np.any(np.abs(theta - theta_old) > max_residual):
    i = rand.random_integers(0, m-1, 1)[0]
    theta_old = theta
    grad_J = np.fromiter([dJ0(theta_old, X[i], Y[i]), dJ1(theta_old, X[i], Y[i])], dtype=np.float)
    theta = theta_old - alpha*grad_J

    # adapt the learning rate by performing a backtracking line search
    while (alpha > 0.0) and ((old_J - J(theta, X[i], Y[i])) <= 0.0):
      alpha = alpha / 2.0
      theta = theta_old - alpha*grad_J
    old_J = J(theta, X[i], Y[i])

    residuals.append(np.abs(theta-theta_old))
    iter_count += 1
  return theta,residuals


# Perform a linear fit using multivariet gradient descent
# with adaptive learning rate
#
# X is a 2D array with rows corresponding to xi and columns
# corresponding to training sets
def fit_line_MGD(X, Y, n=2, plot=False):
  m = len(X)
  x0s = np.ones(m)
  X = np.vstack((x0s, X))
  """
  # scale X
  means = np.fromiter([xi.mean() for xi in X], dtype=np.float)
  means[0] = 0
  stddevs = np.fromiter([xi.std() for xi in X], dtype=np.float)
  stddevs[0] = 1
  #X = (X - means) * stddevs.reciprocal()
  X = ((X.transpose() - means) * np.reciprocal(stddevs)).transpose()
  print X
  """

  def J(thetas, xis, ys):
    return np.sum((np.dot(thetas, xis)-ys)**2)/(2*m)
  def dJi(thetas, xis, ys, i):
    return np.sum((np.dot(thetas, xis)-ys)*xis[i])/m

  alpha = 1.0e9  # initial learning rate
  max_residual = 1e-6
  theta_old = np.zeros(2)
  theta = np.ones(2)
  residuals = []
  iter_count = int(0)
  old_J = J(theta_old, X, Y)
  while np.any(np.abs(theta - theta_old) > max_residual):
    theta_old = theta
    grad_J = np.fromiter([dJi(theta_old, X, Y, i) for i in range(len(theta_old))], dtype=np.float)
    theta = theta_old - alpha*grad_J

    # adapt the learning rate by performing a backtracking line search
    while (alpha > 0.0) and ((old_J - J(theta, X, Y)) <= 0.0):
      alpha = alpha / 2.0
      theta = theta_old - alpha*grad_J
    old_J = J(theta, X, Y)
    #print 'alpha:', alpha
    if plot:
      plt.scatter([iter_count], [J(theta, X, Y)])
      plt.show()

    residuals.append(np.abs(theta-theta_old))
    iter_count += 1
  return theta,residuals

# Perform a multivariet linear fit using SciPy's Conjugate Gradient Descent
#
# X is a 2D array with rows corresponding to xi and columns
# corresponding to training sets
def fit_line_CGD(X, Y, n=2, plot=False):
  m = len(X)
  x0s = np.ones(m)
  X = np.vstack((x0s, X))
  def h(thetas):
    return np.dot(thetas, X)
  # params: Theta, X, y
  def J(thetas):
    return np.sum((h(thetas)-Y)**2)/(2*m)
  def dJdx(thetas):
    return np.sum((h(thetas)-Y)*X, axis=1)/m
  initial_thetas = np.zeros(n)
  return op.minimize(J, initial_thetas, method='Newton-CG', jac=dJdx,
                     options={'xtol': 1e-8, 'disp': True}).x


def sigmoid(z):
  return 1.0 / (1.0 + np.exp(-z))

# Setting the sigmoid function to 0.5 inverting yields the condition
# np.dot(theta.transpose() * X) == 0, where X is a column vector with
# elements x_i. Solving for x_{n-1} yields the formula below
#
# Length of x must must be one less than length of theta
def logistic_regression_boundary(x, Theta):
  print('x shape: {}'.format(np.shape(x)))
  print('Theta shape: {}'.format(np.shape(Theta)))
  return -np.dot(x, Theta[:-1]) / Theta[-1]

# Perform a multivariate linear fit using SciPy's Conjugate Gradient Descent
#
# X is a 2D array with rows corresponding to training sets and columns to xi
# (including x0)
# Y is a 1D column vector (m x 1, m = # training sets)
def logistic_regression(X, Y, lam=1, xtol=1e-8, maxiter=1000):
  def h(thetas):
    z = np.dot(X, thetas)[:,np.newaxis]
    return sigmoid(z)
  def J(thetas):
    J = -np.sum(Y*np.log(h(thetas))+(1-Y)*np.log(1.0-h(thetas)))/m \
      + lam*np.sum(thetas[1:]**2) / (2*m)
    print(np.shape(h(thetas)))
    print(np.shape(Y))
    print(np.shape(X))
    dJdx = np.sum((h(thetas)-Y)*X, axis=0)/m \
         + lam*np.sum(thetas[1:])/m
    print(J)
    return [J, dJdx]
  print(np.shape(X))
  print(np.shape(Y))
  m = X.shape[0]
  n = X.shape[1]
  '''
  n = X.shape[1]+1
  x0s = np.ones((m,1))
  X = np.hstack((x0s, X))
  '''
  initial_thetas = np.zeros(n)
  print('X: {}'.format(X))
  print('Y: {}'.format(Y))
  print('theta: {}'.format(initial_thetas))
  #method = 'Newton-CG'
  method = 'BFGS'
  return op.minimize(J, initial_thetas, method=method, jac=True,
                     options={'disp': True, 'maxiter': maxiter}).x[:,np.newaxis]
                     #options={'xtol': xtol, 'disp': True, 'maxiter': maxiter}).x[:,np.newaxis]
