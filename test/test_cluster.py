#!/usr/bin/python

import numpy as np
import pgl.labview as lv
import pgl.cluster as cluster

lv_dir = '/home/lane/Data/HPRF/2015-09-11/'
lv_file = 'raw_data_2015-09-11@11_13_20.430.npz'

(times, signals) = lv.load_data(lv_dir, lv_file, channels=[2,3,4])
cluster_table = cluster.hierarchical_peak_cluster(times[:2500], signals[0,:2500])
