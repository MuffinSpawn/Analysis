#!/usr/bin/python

import json;
import numpy as np  # NumPy (multidimensional arrays, linear algebra, ...)
import scipy as sp  # SciPy (signal and image processing library)

import os.path
import sys
import re
from bisect import bisect_left, bisect_right

class ParseMode():
  text = 0
  header = 1
  data = 2

class FieldComponent():
  Ez = 0
  Er = 1
  E = 2
  H = 3

class TableColumn():
  Z = 0
  R = 1
  Ez = 2
  Er = 3
  E = 4
  H = 5

class Limits():
  min = 0.0
  max = 0.0

class FieldMap:
  def __init__(self, rows, columns, z_limits, r_limits, delta_z, delta_r):
    self.data_ = np.empty((4, rows, columns))
    self.z_limits_ = z_limits
    self.r_limits_ = r_limits
    self.delta_z_ = delta_z
    self.delta_r_ = delta_r
    self.z_values_ = np.linspace(self.z_limits_.min, self.z_limits_.max,
                                 rows)
    self.r_values_ = np.linspace(self.r_limits_.min, self.r_limits_.max,
                                 columns)

  def z_values(self):
    return self.z_values_

  def r_values(self):
    return self.r_values_

  def Ez(self, z, r):
    z_index = bisect_left(self.z_values_, z)
    r_index = bisect_left(self.r_values_, r)
    return self.data_[FieldComponent.Ez][z_index][r_index]

  def Er(self, z, r):
    z_index = bisect_left(self.z_values_, z)
    r_index = bisect_left(self.r_values_, r)
    return self.data_[FieldComponent.Er][z_index][r_index]

  def E(self, z, r):
    z_index = bisect_left(self.z_values_, z)
    r_index = bisect_left(self.r_values_, r)
    return self.data_[FieldComponent.E][z_index][r_index]

  def H(self, z, r):
    z_index = bisect_left(self.z_values_, z)
    r_index = bisect_left(self.r_values_, r)
    return self.data_[FieldComponent.H][z_index][r_index]


"""
  Load the interpolated field data output from SF7. Parse the field component
  values table and put it in a NumPy array. The first index is the field
  component {FieldComponent.Ez, FieldComponent.Er, FieldComponent.H}. The
  second index is z. The third index is r.

  Return a tuple containing the field map,
"""
def parse_sf7(file):
  z_limits = Limits()
  r_limits = Limits()
  delta_z = 0.0
  delta_r = 0.0
  parse_mode = ParseMode.text
  table_rows = 0
  row = int(0)
  f = open(file, 'r')
  for line in f:
    # print line
    if (parse_mode == ParseMode.text):
      if (line.find("(Zmin,Rmin)") == 0):
        limits = map(float, re.findall(r'-?\d+.?\d*', line))
        # print limits
        z_limits.min = limits[0]
        r_limits.min = limits[1]
      elif (line.find("(Zmax,Rmax)") == 0):
        limits = map(float, re.findall(r'-?\d+.?\d*', line))
        # print limits
        z_limits.max = limits[0]
        r_limits.max = limits[1]
        delta_z = z_limits.max - z_limits.min
        delta_r = r_limits.max - r_limits.min
      elif (line.find("Z and R increments:") == 0):
        parse_mode = ParseMode.header
        # Parse Z and R increments
        increments = map(int, re.findall(r'\d+', line))
        # print increments
        # Calculate sizes of numpy arrays from increments
        # rows = (Zs+1) * (Rs+1)
        table_rows = (increments[0]+1)*(increments[1]+1)
        field_map = FieldMap(increments[0]+1, increments[1]+1,
                             z_limits, r_limits, delta_z, delta_r)
    elif (parse_mode == ParseMode.header):
      # look for (cm) header units -> ParseMode.data
      if (line.find(r'(cm)') > 0):
        parse_mode = ParseMode.data
    elif (parse_mode == ParseMode.data):
      # parse table data and put in numpy arrays
      entries = map(float, re.findall(r'-?\d+.?\d*(?:[Ee][-+]\d+)?', line))
      # print entries
      z = round((entries[TableColumn.Z] - z_limits.min)
              / (delta_z / increments[TableColumn.Z]))
      r = round((entries[TableColumn.R] - r_limits.min)
              / (delta_r / increments[TableColumn.R]))
      # print "".join(("(", str(z), ", ", str(r), ")"))
      field_map.data_[FieldComponent.Ez][z][r] = entries[TableColumn.Ez]
      field_map.data_[FieldComponent.Er][z][r] = entries[TableColumn.Er]
      field_map.data_[FieldComponent.E][z][r] = entries[TableColumn.E]
      field_map.data_[FieldComponent.H][z][r] = entries[TableColumn.H]
      row += 1
      # print entries
      if (row == table_rows):
        parse_mode = ParseMode.text
        # print field_map.data
        #print field_map.data[FieldComponent.H][100][100]
        # print field_map.data_[FieldComponent.H]
  f.close()
  # print field_map.data_[FieldComponent.E][:,-1]
  return field_map