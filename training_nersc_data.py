from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from scipy.cluster.hierarchy import fcluster
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist
import numpy as np
import random
import torch
import torch.nn as nn
import csv
import math
import sys
from sklearn import metrics
from training_lib import train_max
from training_lib import split_train_test
from training_lib import nominal_to_int
from training_lib import format_data
from training_lib import split_by_class
from training_lib import filter_fatal_index
import h5py
import os
from os import listdir
from os.path import isfile, join

def read_data(directory):
    attributes = set()
    result = {}
    files = [join(directory,f) for f in listdir(directory) if isfile(join(directory, f))]
    for data_file in files:
        row = []
        result[data_file] = row
        with open(data_file) as f:
            for line in f:
                elements = line.split('|')
                if len(elements) < 6:
                    continue
                elements = [e.replace(" ", "") for e in elements]
                elements = [e.replace("\n", "") for e in elements]
                attributes.add(elements[3])
                date = elements[1].split('/')
                elements[1] = date[2] + date[0] + date[1]
                row.append(elements[1:len(elements)])
    attributes = list(attributes)
    attributes.sort()
    return result, attributes

def cluster_events(data,attributes):
    result = {}
    if len(data) < 1:
        return result
    date = data[0][0]
    table = {}
    count = 0
    for attribute in attributes:
        table[attribute] = 0
    for i in range(0,len(data)):
        if data[i][0] == date:
            table[data[i][2]] += 1
            count += 1
        else:
            v = np.zeros(len(attributes))
            for j in range(0,len(attributes)):
                v[j] = (table[attributes[j]] + .0)/count
                table[attributes[j]] = 0
            result[date] = v
            date = data[i][0]
            count = 1
            table[data[i][2]] = 1
    return result

if __name__ == "__main__":
    data, attributes = read_data('bebop_data')
    #print(data.keys())
    dates = set()
    results = {}
    for node in data:
        node_result = cluster_events(data[node],attributes)
        results[node] = node_result
        for date in node_result:
            dates.add(date)
    failure_proportion = {}
    dates = list(dates)
    dates.sort()
    for date in dates:
        failure_proportion[date] = 0
    for node in results:
        for date in results[node]:
            failure_proportion[date] += 1
    for date in dates:
        #failure_proportion[date] /= (.0+len(results))
        print("date: {0}, failure_proportion = {1}".format(date,failure_proportion[date]))
        failure_proportion[date] /= (.0+len(results))
    print("total failure dates = {0}, critical failures = {1}".format(len(dates),len([date for date in failure_proportion if failure_proportion[date]>.5])))
