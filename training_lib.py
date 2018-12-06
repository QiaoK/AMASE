from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from scipy.cluster.hierarchy import fcluster
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist
import numpy as np
import random
import csv
import math
import sys
from sklearn import metrics
train_max=1462075200
def nominal_to_int(data_y,table):
    '''
    result=np.zeros([len(data_y),len(table)])
    for i in range(0,len(data_y)):
        result[i,table[data_y[i]]]=1
    return result
    '''
    return np.asarray(data_y)

def format_data(data,entries):
    feature_names=[]
    conv_names=[]
    for k in entries:
        if 'TEMPORAL_INTERVAL' in k or 'SPATIAL_INTERVAL' in k:
            conv_names.append(k)
        else:
            feature_names.append(k)
    conv_names.sort()
    features=np.empty((len(data),len(feature_names)))
    if len(conv_names)>0:
        conv_features=np.empty((len(data),len(conv_names)))
    for i in range(len(data)):
        for j in range(0,len(feature_names)):
            features[i,j]=data[i][feature_names[j]]
        count1=0
        for j in range(0,len(conv_names)):
            conv_features[i,count1]=data[i][conv_names[j]]
            count1+=1
    if len(conv_names)>0:
        return features,conv_features
    else:
        return features,None

def split_by_class(data_x,data_y,dates,table):
  result={}
  for t in table:
    result[table[t]]=[[],[],[]]
  for i in range(len(data_y)):
    result[table[data_y[i]]][0].append(data_x[i])
    result[table[data_y[i]]][1].append(data_y[i])
    result[table[data_y[i]]][2].append(dates[i])
  return result

def split_train_test(data_x,data_y,dates,fatal_dates,train_max):
  data_x_train=[]
  data_y_train=[]
  data_x_test=[]
  data_y_test=[]
  test_dates=[]
  fatal_dict={}
  fatal_clusters=set()
  for i in range(len(data_x)):
    if data_y[i]==1:
      fatal_clusters.add(fatal_dates[i])
    if dates[i]<=train_max:
      data_x_train.append(data_x[i])
      data_y_train.append(data_y[i])
    else:
      data_x_test.append(data_x[i])
      data_y_test.append(data_y[i])
      test_dates.append(dates[i])
      if data_y[i]==1:
        if dates[i] not in fatal_dict:
          fatal_dict[dates[i]]=set()
        fatal_dict[dates[i]].add(fatal_dates[i])
  print("train:test={0}:{1},{2},total fatal clusters={3}".format(len(data_x_train),len(data_x_test),(0.0+len(data_x_train))/len(data_x),len(fatal_clusters)))
  return data_x_train,data_y_train,data_x_test,data_y_test,test_dates,fatal_dict
