"""Short and sweet LSTM implementation in Tensorflow.
Motivation:
When Tensorflow was released, adding RNNs was a bit of a hack - it required
building separate graphs for every number of timesteps and was a bit obscure
to use. Since then TF devs added things like `dynamic_rnn`, `scan` and `map_fn`.
Currently the APIs are decent, but all the tutorials that I am aware of are not
making the best use of the new APIs.
Advantages of this implementation:
- No need to specify number of timesteps ahead of time. Number of timesteps is
  infered from shape of input tensor. Can use the same graph for multiple
  different numbers of timesteps.
- No need to specify batch size ahead of time. Batch size is infered from shape
  of input tensor. Can use the same graph for multiple different batch sizes.
- Easy to swap out different recurrent gadgets (RNN, LSTM, GRU, your new
  creative idea)
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from scipy.cluster.hierarchy import fcluster
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist
import numpy as np
import random
import tensorflow as tf
import tensorflow.contrib.layers as layers
import csv
import math
import sys

map_fn = tf.map_fn

def nominal_to_binary(data_y):
    nominals=list(set(data_y))
    size=len(nominals)
    output_bits=int(math.ceil(math.log(size,2)))
    table={}
    result=[]
    print(output_bits)
    for i in range(0,size):
        list_form=as_bytes(i, output_bits)
        pattern=",".join(str(x) for x in list_form)
        table[nominals[i]]=pattern
    for y in data_y:
        bits=table[y].split(",")
        bits=[int(b) for b in bits]
        result.append(bits)
    return result,table

def create_nominal_table(data_y):
    nominals=list(set(data_y))
    size=len(nominals)
    table={}
    for i in range(0,size):
      table[nominals[i]]=i
    return table

def nominal_to_int(data_y,table):
    result=np.zeros([len(data_y),len(table)])
    for i in range(0,len(data_y)):
        result[i,table[data_y[i]]]=1
    return result

# Parameters
learning_rate = 0.001
batch_size = 1024
display_step = 1
output_size = 2

# Network Parameters

# Create model
def multilayer_perceptron(x,weights,biases):
  # Hidden fully connected layer with 256 neurons
  #layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
  # Hidden fully connected layer with 256 neurons
  #layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
  # Output fully connected layer with a neuron for each class
  #out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
  dense = tf.layers.dense(inputs=x, units=256, activation=tf.nn.relu)
  dropout = tf.layers.dropout(
      inputs=dense, rate=0.41)
  dense2 = tf.layers.dense(inputs=dropout, units=512, activation=tf.nn.relu)
  out_layer=tf.layers.dense(inputs=dense2, units=output_size)

  return out_layer


def split_by_class(data_x,data_y,table):
  result={}
  for t in table:
    result[table[t]]=[[],[]]
  for i in range(len(data_y)):
    result[table[data_y[i]]][0].append(data_x[i])
    result[table[data_y[i]]][1].append(data_y[i])
  return result


def format_data(data,entries):
  result=np.empty([len(data),len(data[0])])
  for i in range(len(data)):
    for j in range(len(entries)):
      result[i,j]=data[i][entries[j]]
  return result

def replicate_class(table,data_x,data_y,percent,name,size):
  new_data_x=[]
  new_data_y=[]
  for label in table:
    if label!=name:
      continue
    count=0
    for y in data_y:
      if y==label:
        count+=1;
    if count>0 and (count+.0)/size<percent:
      index=[]
      for i in range(len(data_y)):
        if data_y[i]==label:
          index.append(i)
      copies=(int)((percent*size+count-1)/count)-1
      print("expanding label {0} to {1} copies {2}".format(label,copies,(count+.0)/size))
      for i in range(copies):
        for j in range(len(index)):
          new_data_x.append(data_x[index[j]])
          new_data_y.append(data_y[index[j]])
  for i in range(len(new_data_x)):
    data_x.append(new_data_x[i])
    data_y.append(new_data_y[i])
  index=[i for i in range(len(data_x))]
  random.shuffle(index)
  #print(index)
  new_data_x=[]
  new_data_y=[]
  for i in range(len(index)):
    new_data_x.append(data_x[index[i]])
    new_data_y.append(data_y[index[i]])
  for i in range(len(index)):
    data_x[i]=new_data_x[i]
    data_y[i]=new_data_y[i]

def clustering(fatal_feature,k):
  fatal_feature_matrix=[]
  entries=None
  with open(fatal_feature) as csvfile:
    reader = csv.DictReader(csvfile)
    for x in reader:
      temp=[]
      if entries==None:
        entries=x.keys()
      for entry in entries:
        temp.append(x[entry])
      fatal_feature_matrix.append(temp)
  dist=pdist(fatal_feature_matrix)
  result=fcluster(linkage(dist, method='single'), k-1, 'maxclust')
  return result

def split_train_test(data_x,data_y,dates,train_max):
  data_x_train=[]
  data_y_train=[]
  data_x_test=[]
  data_y_test=[]
  for i in range(len(data_x)):
    if dates[i]<=train_max:
      data_x_train.append(data_x[i])
      data_y_train.append(data_y[i])
    else:
      data_x_test.append(data_x[i])
      data_y_test.append(data_y[i])
  print("train:test={0}:{1},{2}".format(len(data_x_train),len(data_x_test),(0.0+len(data_x_train))/len(data_x)))
  return data_x_train,data_y_train,data_x_test,data_y_test

if __name__ == "__main__":
  # Load training and eval data
  train_max=1462075200
  training_epochs=int(sys.argv[2])
  tf.set_random_seed(5555)
  data_x=[]
  data_y=[]
  dates=[]
  #cluster_membership=clustering(sys.argv[2],output_size)
  #print("Cluster number",len(set(cluster_membership)))
  fatals=0
  with open(sys.argv[1]) as csvfile:
    reader = csv.DictReader(csvfile)
    for x in reader:
      #del x[""]
      #del x["START"]
      #del x["END"]
      temp=int(x["FATAL"])
      if temp!=1:
        temp=0
      if temp==1:
        fatals+=1
      #if temp>0:
      #  temp=cluster_membership[temp-1]
      data_y.append(temp)
      dates.append(int(x["DATE"]))
      del x["FATAL"]
      del x["DATE"]
      del x["LEAD_TIME"]
      data_x.append(x)
  entries=data_x[0].keys()
  table=create_nominal_table(data_y)
  data_x_train=data_x[0:int(len(data_x)*0.8)]
  data_y_train=data_y[0:int(len(data_x)*0.8)]
  data_x_test=data_x[int(len(data_x)*0.8+1):len(data_x)]
  data_y_test=data_y[int(len(data_y)*0.8+1):len(data_y)]
  #data_x_train,data_y_train,data_x_test,data_y_test=split_train_test(data_x,data_y,dates,train_max)
  print("fatal data size is {0}".format(fatals))
  print("original train data size={0}".format(len(data_y_train)))
  replicate_class(table,data_x_train,data_y_train,.5,1,len(data_y_train))
  replicate_class(table,data_x_train,data_y_train,.5,0,len(data_y_train))
  print("new train data size={0}".format(len(data_y_train)))
  data_by_class = split_by_class(data_x_train,data_y_train,table)
  print("train 0 has {0},1 has {1}".format(len(data_by_class[0][1]),len(data_by_class[1][1])))
  train_data = format_data(data_x_train,entries)
  train_labels = nominal_to_int(data_y_train,table)
  eval_data = format_data(data_x_test,entries)
  eval_labels = nominal_to_int(data_y_test,table)
  data_by_class = split_by_class(data_x_test,data_y_test,table)
  eval_data0 = format_data(data_by_class[0][0],entries)
  eval_label0 = nominal_to_int(data_by_class[0][1],table)
  eval_data1 = format_data(data_by_class[1][0],entries)
  eval_label1 = nominal_to_int(data_by_class[1][1],table)
  print("eval 0 has {0}, 1 has {1}".format(len(data_by_class[0][0]),len(eval_data1)))
  regr = linear_model.LinearRegression()
  regr.fit(train_data, train_labels)
  
  predict_y = regr.predict(eval_data)
  print('Coefficients: \n', regr.coef_)
  count=0
  for i in range(len(eval_labels)):
    maximum_predict=0
    maximum_label=0
    predict=0
    label=0
    for j in range(len(table)):
      if maximum_predict<predict_y[i,j]:
        maximum_predict=predict_y[i,j]
        predict=j
      if maximum_label<eval_labels[i,j]:
        maximum_label=eval_labels[i,j]
        label=j
    if label==predict:
      count+=1
  print(count/len(predict_y))

  n_input = len(entries)
  n_classes = len(table)
  n_hidden_1 = 256 # 1st layer number of neurons
  n_hidden_2 = 256 # 2nd layer number of neurons
# tf Graph input
  X = tf.placeholder("float", [None, n_input])
  Y = tf.placeholder("float", [None, n_classes])

# Store layers weight & bias
  weights = {
      'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
      'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
      'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
  }
  biases = {
      'b1': tf.Variable(tf.random_normal([n_hidden_1])),
      'b2': tf.Variable(tf.random_normal([n_hidden_2])),
      'out': tf.Variable(tf.random_normal([n_classes]))
  }

  # Construct model
  logits = multilayer_perceptron(X,weights,biases)

  # Define loss and optimizer
  loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=logits, labels=Y))
  optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
  train_op = optimizer.minimize(loss_op)

  with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    # Test model
    pred = tf.nn.softmax(logits)  # Apply softmax to logits
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(Y, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    # False negative
    report_event = tf.equal(tf.cast(tf.constant(1),"int64"), tf.argmax(pred, 1))
    false_negative_percent = tf.reduce_mean(tf.cast(report_event,"float"))
    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(len(train_data)/batch_size)
        # Loop over all batches
        for i in range(total_batch):
            batch_x = train_data[(i*batch_size):((i+1)*batch_size)]
            batch_y = train_labels[(i*batch_size):((i+1)*batch_size)]
            #print(len(batch_x[0]))
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([train_op, loss_op], feed_dict={X: batch_x,
                                                            Y: batch_y})
            # Compute average loss
            avg_cost += c / total_batch
        # Display logs per epoch step
        if epoch % display_step == 0:
          fn_count = false_negative_percent.eval({X: eval_data0, Y: eval_label0})*len(eval_label0)
          tp=accuracy.eval({X: eval_data1, Y: eval_label1})
          recall=tp*len(eval_data1)/(tp*len(eval_data1)+fn_count)
          print("Epoch:", '%04d' % (epoch+1), "cost={:.9f},accuracy={:.4f},precision={:.4f},recall={:.4f}".format(avg_cost,accuracy.eval({X: eval_data, Y: eval_labels}),tp,recall))
    print("Optimization Finished!")
    #saver = tf.train.Saver()
    #save_path = saver.save(sess, "/home/qkt561/tensor_model.ckpt")
    print(table)
    print("Accuracy:", accuracy.eval({X: eval_data, Y: eval_labels}))
    tp=accuracy.eval({X: eval_data1, Y: eval_label1})
    print("Accuracy label 1:",tp)
    tn1=accuracy.eval({X: eval_data0, Y: eval_label0})
    print("Accuracy label 0:", tn1)
    fn_count = false_negative_percent.eval({X: eval_data0, Y: eval_label0})*len(eval_label0)
    print("total non-fatal features are {0}, fn_count={1}".format(len(eval_data0),fn_count))
    print("Precision={0},recall={1}".format(tp,tp*len(eval_data1)/(tp*len(eval_data1)+fn_count)))
