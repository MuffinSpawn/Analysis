# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 07:42:15 2015

@author: lane
"""
#!/usr/bin/python

import json;
import numpy as np  # NumPy (multidimensional arrays, linear algebra, ...)
import numpy.linalg as linalg
import scipy as sp  # SciPy (signal and image processing library)

import os.path
import sys
import re
import math
from bisect import bisect_left, bisect_right

def counting_sort(A):
  k = np.max(A)
  C = np.zeros(k+1, dtype=int)
  for index in range(len(A)):
    C[A[index]] += 1
  for index in range(1, k+1):
    C[index] += C[index-1]
  B = np.empty(len(A), dtype=int)
  for index in range(len(A)-1, -1, -1):
    B[C[A[index]]-1] = A[index]
    C[A[index]] -= 1
  return B

some_ints = np.array([3, 6, 1, 2, 3, 0, 9, 4, 5, 8, 8, 3, 4], dtype=int)
print some_ints
print counting_sort(some_ints)
