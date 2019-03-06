import math
import json;
import numpy as np  # NumPy (multidimensional arrays, linear algebra, ...)
import numpy.linalg as linalg
import scipy as sp  # SciPy (signal and image processing library)

import os.path
import sys
import re
import math
from bisect import bisect_left, bisect_right
import pgl.curve as curve

# Group the X values into k clusters using a distance function that measures
# the angular distance between the centroid vertical and the points.
def kmeans_peak_cluster(X, Y, K=2):
  p = 2  # number of dimensions
  cluster_assignments = np.zeros(len(X))
  # randomly assign points to one of the clusters
  new_cluster_assignments = np.random.random_integers(0, K-1, len(X))
  # loop until all the cluster assignments remain unchanged between iterations
  # FIXME: detect and eliminate deadlock due to waffling between two cluster assignments
  sift_count = 0
  while sift_count < K**2 and \
        np.sum(cluster_assignments - new_cluster_assignments) != 0:
    diff =  np.where(cluster_assignments != new_cluster_assignments)[0]
    cluster_assignments = new_cluster_assignments.copy()
    # print cluster_assignments

    # calculate the centroids (vector of mean coordinates of the cluster
    # members)
    centroids = np.zeros((K,p))
    cluster_sizes = np.zeros((K))
    for i in range(len(X)):
      k = cluster_assignments[i]
      centroids[k,0] += X[i]
      centroids[k,1] += Y[i]
      cluster_sizes[k] += 1
    for k in range(K):
      centroids[k] = centroids[k] / cluster_sizes[k]
    # print centroids

    # reassign each point to the cluster whose centroid is closest
    objective = 0
    for i in range(len(X)):
      centroid_distances = np.zeros((K))
      for k in range(K):
        # Calculate the angular distance between a vertical line going
        # through the centroid and the point To find this angle we
        # subtract off the x coordinate of the centroid. This yields a
        # y-axis vector (x_v) and a vector pointing away from the origin
        # at the same angle as that of the point away from the centroid
        # vertical (x_p). Starting with the formula for the dot product
        # of x_p and x_v we can find the 2nd order Taylor approximation
        # of cos(theta), solve for theta, and then Taylor approximate
        # the root to 1st order to get
        # theta ~= 1-x_p.x_v'/sqrt((x_p.x_p')(x_v.x_v')). This is called
        # the Cosine Difference
        # (http://arxiv.org/ftp/arxiv/papers/1405/1405.7471.pdf).
        x_p = np.array((X[i]-centroids[k,0], Y[i]))
        x_v = np.array((0, centroids[k,0]))
        centroid_distances[k] = \
          1 - np.dot(x_p,x_v)/math.sqrt(np.dot(x_p,x_p)*np.dot(x_v,x_v))
      objective += centroid_distances.min()
      new_cluster_assignments[i] = centroid_distances.argmin()
    sift_count += 1
  #print ''.join(('K-means Sift Count: ', str(sift_count)))
  return (new_cluster_assignments, centroids, objective)

def add_cluster(cluster_table, cluster1_id, cluster2_id):
  new_id = 0
  if len(cluster_table) > 0:
    new_id = max(cluster_table.keys())+1
  cluster1 = cluster_table[cluster1_id]
  cluster2 = cluster_table[cluster2_id]
  mass = cluster1[1] + cluster2[1]
  x = (cluster1[0]*cluster1[1]+cluster2[0]*cluster2[1]) / mass
  force = cluster1[1]*cluster2[1]/(cluster1[0]-cluster2[0])**2
  entry = np.array((x, mass, cluster1_id, cluster2_id, force))
  cluster_table[new_id] = entry

def add_force(force_table, cluster_table, cluster1_id, cluster2_id, row):
  cluster1 = cluster_table[cluster1_id]
  cluster2 = cluster_table[cluster2_id]
  force = cluster1[1]*cluster1[1]/(cluster1[0]-cluster2[0])**2
  force_entry = np.array((force, cluster1_id, cluster2_id))
  # Assuming the force table is already sorted, insert the new entry such
  # that we keep the table sorted
  """
  insertion_index = 0
  if len(force_table) > 0:
    insertion_index = bisect_left(force_table[:,0], force)
  force_table = np.insert(force_table, insertion_index, force_entry, axis=0)
  return force_table
  """
  force_table[row] = force_entry

# Hierarchical cluster algorithm using an inverse square law distance measure.
# distance = y1*y2/(x1-x2)^2
# Create a table for clusters:
#   Cluster ID, X, Mass, Cluster1 ID, Cluster2 ID, Force
# - Populate initially with points where Cluster ID = x-axis index,
#   Cluster1 ID = Cluster2 ID = Distance = -1
# Create a sorted table for forces:
#   Force, Cluster1 ID, Cluster2 ID
# - Populate initially with distances between each unique set of points
# - Update with each newly created cluster
#   - Remove entries containing the component subclusters
#   - Add entries between the new cluster and all the remaining clusters
def hierarchical_peak_cluster_slow(X, Y):
  # Initialize the cluster table
  #       (Cluster ID, Subcluster1 ID, Subcluster2 ID, Force)
  cluster_table = {}
  cluster_id = 0
  print('Initializing cluster table...')
  for x,y in zip(X,Y):
    if len(cluster_table) > 0:
      cluster_id = max(cluster_table.keys()) + 1
    cluster_entry = np.array((x, y, -1, -1, 0))
    cluster_table[cluster_id] = cluster_entry
  #print cluster_table

  # Initialize the force table with the forces between each individual point
  N = len(X)
  initial_size = math.floor(N/2.0)*(N-1)
  if N%2 > 0:
    initial_size += (N-1)/2
  force_table = np.empty((initial_size,3))
  print('Initializing force table...')
  row = int(0)
  for i,cluster1_id in enumerate(cluster_table.keys()):
    for cluster2_id in cluster_table.keys()[i+1:]:
      add_force(force_table, cluster_table, cluster1_id, cluster2_id, row)
      row += 1
  # Sort the force table by force from weakest to strongest
  force_table = sorted(force_table, key=lambda entry:entry[1])
  print(force_table)
  return None

  # Iterate through the force table creating a cluster from the two
  # sub-clusters that have the strongest force between them. To shrink the force
  # table we must remove elements related to the sub-clusters.
  force_entry = force_table[-1]
  print('Clustering...')
  while len(force_table) > 0:
    print(''.join(('Creating cluster with ID ', str(max(cluster_table.keys())+1))))
    # Create a new cluster using the two sub-clusters for which the force
    # between them is the last entry in the force table (largest force).
    add_cluster(cluster_table, force_entry[1], force_entry[2])

    # Remove the last row
    delete = {len(force_table)-1:True}

    # Remove rows that contain cluster IDs for the two sub-clusters and,
    # meanwhile, make a list of the cluster IDs still represented in the table
    remaining = {}
    for index,entry in force_table[1:]:
      if (entry[1] == force_entry[1]) or (entry[1] == force_entry[2]) or \
         (entry[2] == force_entry[1]) or (entry[2] == force_entry[2]):
        delete_indicies[index] = True
      else:
        remaining[entry[1]] = True
        remaining[entry[2]] = True
    force_table.delete(delete.keys())

    # Insert entries for the forces between the new cluster and all the others
    # in the table.
    cluster1_id = max(cluster_table.keys())
    for cluster2_id in remaining.keys():
      force_table = add_force(force_table, cluster_table, cluster1_id, cluster2_id)
  force_entry = force_table[-1]
  return cluster_table

def cluster_xy(X, Y, cluster):
  total_mass = 0.0
  temp_sum = 0.0
  for index in cluster:
    total_mass += Y[index]
    temp_sum += X[index]*Y[index]
  return (temp_sum / total_mass, total_mass)

def cluster_force(X, Y, cluster_xy, point_index):
  cx = cluster_xy[0]
  cy = cluster_xy[1]
  x = X[point_index]
  y = Y[point_index]
  return cy*y/(cx-x)**2

def point_force(X, Y, point1_index, point2_index):
  if (point1_index >= len(X)) or (point2_index >= len(X)):
    return 0.0
  x1 = X[point1_index]
  y1 = Y[point1_index]
  x2 = X[point2_index]
  y2 = Y[point2_index]
  return y1*y2/(x1-x2)**2

def hierarchical_peak_cluster(X, Y):
  # Cluster table: cluster_id: array of point indicies
  cluster_table = {0:[0]}
  xy_table = {0:[X[0], Y[0]]}
  next_cluster_id = 1
  for index in range(1,len(X)):
    #print ''.join(('Point: ', str(index)))
    max_force = point_force(X, Y, index, index+1)
    strongest_attractor = -1
    for cluster_id in cluster_table.keys():
      cluster = cluster_table[cluster_id]
      # print ''.join(('cluster_id: ', str(cluster_id)))
      # print ''.join(('cluster: ', str(cluster)))
      force = cluster_force(X, Y, xy_table[cluster_id], index)
      if force > max_force:
        max_force = force
        strongest_attractor = cluster_id
    # print ''.join(('strongest_attractor: ', str(strongest_attractor)))
    if strongest_attractor >= 0:
      # Add to an existing cluster
      cluster = cluster_table[cluster_id]
      cluster.append(index)
      cluster_table[cluster_id] = cluster
      xy_table[cluster_id] = cluster_xy(X, Y, cluster)
    else:
      # Create a new cluster
      cluster_table[next_cluster_id] = [index]
      xy_table[next_cluster_id] = [X[index], Y[index]]
      next_cluster_id += 1
    print(xy_table)
    # print ''.join(('cluster_table: ', str(cluster_table)))
  # Time to beat:  0m14.370s
  for cluster_id in cluster_table.keys():
    print(''.join((str(cluster_id), ': ', str(cluster_table[cluster_id]))))

# Use a k-means algorithm and Gaussian fits to find the spectral peak clusters.
# Steps:
#   1) K=2
#   Repeat until the SSR minimum is found:
#   2) Run k-means algorithm several times with the same value of K to get the lowest objective score
#   3) Fit a Gaussian to each cluster and calculate the total sum of the squared residuals (SSR)
#   4) K += 1
def spectral_cluster(frequencies, magnitudes, krange=(8,10)):
    if (len(frequencies) < 2):
        return None
    repetitions = 5
    max_mag_freq = frequencies[magnitudes.argmax()]
    initial_fit = curve.fit_gaussian_with_mean(frequencies, magnitudes, mu=max_mag_freq, yoff=False)
    lowest_ssr = curve.gaussian_ssr(initial_fit, frequencies, magnitudes)
    ssrs = np.zeros(11)
    #last_ssr = ssr+1
    #K = 2
    # Repeat until the SSR minimu is found
    gaussian_fits = np.empty((2, 4))
    best_gaussian_fits = np.empty((2,4))
    for K in range(krange[0], krange[1]):
        #last_ssr = ssr
        # FIXME: run several times for best objective value
        best_clustering_results = kmeans_peak_cluster(frequencies, magnitudes, K)
        for index in range(1,10):
            clustering_results = kmeans_peak_cluster(frequencies, magnitudes, K)
            print(''.join(('Clustering Objective: ', str(clustering_results[2]))))
            if clustering_results[2] < best_clustering_results[2]:
                best_clustering_results = clustering_results
        print(''.join(('Best Clustering Objective: ', str(best_clustering_results[2]))))
        #cluster_assignments, centroids, objective = kmeans_peak_cluster(frequencies, magnitudes, K)
        cluster_assignments, centroids, objective = best_clustering_results
        cluster_start = 0
        next_cluster_start = 0
        #ssr = 0
        gaussian_fits = np.zeros((K, 4))
        # Fit a Gaussian to each cluster and calculate the total sum of the squared residuals (SSR)
        for k in range(K):
            cluster_indicies = np.where(cluster_assignments==k)[0]
            # print ''.join(('Cluster Start: ', str(cluster_indicies[0]), '\tCluster End: ', str(cluster_indicies[-1])))
            cluster_freqs = frequencies[cluster_indicies[0]:cluster_indicies[-1]+1]
            cluster_mags = magnitudes[cluster_indicies[0]:cluster_indicies[-1]+1]
            max_mag_index = cluster_mags.argmax()
            max_mag_freq = cluster_freqs[max_mag_index]
            print(max_mag_freq)
            weights = np.random.normal(0.1, 0.9, len(cluster_mags))  # purely emperical choices for mu and sigma here
            weights[max_mag_index] = 0.01                            # ^ ditto ^
            gaussian_fits[k] = curve.fit_gaussian_with_mean(cluster_freqs, cluster_mags, mu=max_mag_freq, yerr=weights, yoff=False)
            ssrs[K] += curve.gaussian_ssr(gaussian_fits[k], cluster_freqs, cluster_mags)
            print(''.join(('ssrs[', str(K), ']: ', str(ssrs[K]))))
        ssrs[K] /= K
        if ssrs[K] < lowest_ssr:
            lowest_ssr = ssrs[K]
            best_gaussian_fits = gaussian_fits
        print(''.join(('lowest_ssr: ', str(lowest_ssr))))
        #K = K + 1
        print(''.join(('K: ', str(K), '\tSSR: ', str(ssrs[K]))))
    print(''.join(('Best K: ', str(np.where(ssrs == lowest_ssr)[0][0]), '\tSSR: ', str(lowest_ssr))))
    return best_gaussian_fits

def kmeans_signal_cluster(signals, K=2):
  signal_length = np.shape(signals)[1]  # signal length
  cluster_assignments = np.zeros(len(signals))
  # randomly assign points to one of the clusters
  new_cluster_assignments = np.random.random_integers(0, K-1, len(signals))
  # loop until all the cluster assignments remain unchanged between iterations
  # FIXME: detect and eliminate deadlock due to waffling between two cluster assignments
  sift_count = 0
  cluster_averages = np.zeros((K,signal_length))
  objective = 0
  while sift_count < K**2 and \
        np.sum(cluster_assignments - new_cluster_assignments) != 0:
    diff =  np.where(cluster_assignments != new_cluster_assignments)[0]
    cluster_assignments = new_cluster_assignments.copy()
    # print cluster_assignments

    cluster_averages = np.zeros((K,signal_length))
    cluster_sizes = np.zeros((K))
    for i in range(len(signals)):
      k = cluster_assignments[i]
      cluster_averages[k] += signals[i]
      cluster_sizes[k] += 1
    for k in range(K):
      if cluster_sizes[k] > 0:
        cluster_averages[k] = cluster_averages[k] / cluster_sizes[k]

    # Reassign each signal to the cluster for which the MSE between each
    # signal and the average cluster signal is the smallest.
    objective = 0
    for i,signal in enumerate(signals):
      cluster_MSEs = np.zeros((K))
      for k in range(K):
        cluster_MSEs[k] = math.sqrt(np.sum((signal-cluster_averages[k])**2)/signal_length)
      #print "Signal", i, "MSEs:", cluster_MSEs
      objective += cluster_MSEs.min()
      new_cluster_assignments[i] = cluster_MSEs.argmin()
    sift_count += 1
  #print ''.join(('K-means Sift Count: ', str(sift_count)))
  return (new_cluster_assignments, cluster_averages, objective)
