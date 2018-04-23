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
training_epochs = 50
batch_size = 32
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
      inputs=dense, rate=0.42)
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

def replicate_class(table,data_x,data_y,percent):
  new_data_x=[]
  new_data_y=[]
  for label in table:
    count=0
    for y in data_y:
      if y==label:
        count+=1;
    if count>0 and (count+.0)/len(data_y)<percent:
      index=[]
      for i in range(len(data_y)):
        if data_y[i]==label:
          index.append(i)
      copies=(int)((percent*len(data_y)+count-1)/count)-1
      print("expanding label {0} to {1} copies {2}".format(label,copies,(count+.0)/len(data_y)))
      for i in range(copies):
        for j in range(len(index)):
          new_data_x.append(data_x[index[j]])
          new_data_y.append(data_y[index[j]])
  for i in range(len(new_data_x)):
    data_x.append(new_data_x[i])
    data_y.append(new_data_y[i])
  index=range(len(data_x))
  random.shuffle(index)
  new_data_x=[]
  new_data_y=[]
  for i in range(len(index)):
    new_data_x.append(data_x[index[i]])
    new_data_y.append(data_y[index[i]])
  for i in range(len(index)):
    data_x[i]=new_data_x[i]
    data_y[i]=new_data_y[i]

def discretize_labels(data_y,threshold):
  count=0
  for i in range(len(data_y)):
    if data_y[i]<=threshold :
      data_y[i]=1
      count+=1
    else:
      data_y[i]=0
  return count

if __name__ == "__main__":
  # Load training and eval data
  tf.set_random_seed(555)
  data_x=[]
  data_y=[]
  with open(sys.argv[1]) as csvfile:
    reader = csv.DictReader(csvfile)
    for x in reader:
      #del x[""]
      #del x["START"]
      #del x["END"]
      temp=int(x["LEAD_TIME"])
      if temp>=0:
        data_y.append(temp)
        del x["FATAL"]
        del x["LEAD_TIME"]
        data_x.append(x)
  entries=data_x[0].keys()
  print(len(data_y))
  quantile=discretize_labels(data_y,7200)
  print("time quantile={0}".format(quantile))
  table=create_nominal_table(data_y)
  data_x_train=data_x[0:int(len(data_x)*0.8)]
  data_y_train=data_y[0:int(len(data_x)*0.8)]
  data_x_test=data_x[int(len(data_x)*0.8+1):len(data_x)]
  data_y_test=data_y[int(len(data_y)*0.8+1):len(data_y)]
  print("original train data size={0}".format(len(data_y_train)))
  replicate_class(table,data_x_train,data_y_train,.5)
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
  print("eval 0 has {0}, 1 has {1}".format(len(eval_data0),len(eval_data1)))
  regr = linear_model.LinearRegression()
  regr.fit(train_data, train_labels)
  
  predict_y = regr.predict(eval_data)
  print('Coefficients: \n', regr.coef_)
  count=0
  for i in range(len(eval_labels)):
    test=True
    for j in range(len(table)):
      test=test and round(predict_y[i,j])==eval_labels[i,j]
    if test:
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
            print("Epoch:", '%04d' % (epoch+1), "cost={:.9f}".format(avg_cost))
    print("Optimization Finished!")
    #saver = tf.train.Saver()
    #save_path = saver.save(sess, "/home/qkt561/tensor_model.ckpt")
    # Test model
    pred = tf.nn.softmax(logits)  # Apply softmax to logits
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(Y, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print(table)
    print("Accuracy:", accuracy.eval({X: eval_data, Y: eval_labels}))
    tf=accuracy.eval({X: eval_data1, Y: eval_label1})
    print("Accuracy label 1:",tf)
    tp=accuracy.eval({X: eval_data0, Y: eval_label0})
    print("Accuracy label 0:", tp)
    print("Precision={0},recall={1}".format(tp,tp*len(eval_data0)/(tp*len(eval_data0)+(1-tf)*len(eval_data1))))
