from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

import numpy as np
import random
import tensorflow as tf
import tensorflow.contrib.layers as layers
import csv
import math
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

class TrainingFold:
    def __init__(self, train_x,train_y,valid_x,valid_y):
        self.train_x=train_x
        self.train_y=train_y
        self.valid_x=valid_x
        self.valid_y=valid_y

# Parameters
learning_rate = 0.001
training_epochs = 30
batch_size = 256
display_step = 1

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
      inputs=dense, rate=0.37)
  dense2 = tf.layers.dense(inputs=dropout, units=256, activation=tf.nn.relu)
  out_layer=tf.layers.dense(inputs=dense2, units=2)

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

def count_fatals(x):
  for k in x:
    if int(x[k])>0:
      return 1
  return 0

def create_features():
  job_fatals={}
  djc={}
  data_y2=[]
  with open('job_fatals.csv') as csvfile:
    reader = csv.DictReader(csvfile)
    for x in reader:
      job_fatals[x['']]=x
      del job_fatals[x['']]['']
  with open('djc.csv') as csvfile:
    reader = csv.DictReader(csvfile)
    for x in reader:
      djc[x['']]=x
      del djc[x['']]['']
  data_x=[]
  data_y1=[]
  for k in djc:
    if k in job_fatals:
      job_fatal=count_fatals(job_fatals[k])
    else:
      job_fatal=0
    if int(djc[k]['EXIT_STATUS'])==0:
      data_y2.append(0)
    else:
      data_y2.append(1)
    del djc[k]['EXIT_STATUS']
    #del djc[k]['REQUESTED_CORE_HOURS']
    data_x.append(djc[k])
    data_y1.append(job_fatal)
  return data_x,data_y1,data_y2
    
def replicate_class(table,data_x,data_y,percent):
  new_data_x=[]
  new_data_y=[]
  for label in table:
    count=0
    for y in data_y:
      if y==label:
        count+=1;
    if (count+.0)/len(data_y)<percent:
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

if __name__ == "__main__":
  # Load training and eval data
  tf.set_random_seed(5555)

  data_x,data_y1,data_y2=create_features()
  data_y=data_y1

  entries=data_x[0].keys()
  table=create_nominal_table(data_y)
  data_x_train=data_x[0:int(len(data_x)*0.8)]
  data_y_train=data_y[0:int(len(data_x)*0.8)]
  data_x_test=data_x[int(len(data_x)*0.8+1):len(data_x)]
  data_y_test=data_y[int(len(data_y)*0.8+1):len(data_y)]
  print("original data size={0}".format(len(data_y_train)))
  replicate_class(table,data_x_train,data_y_train,.35)
  print("new data size={0}".format(len(data_y_train)))
  data_by_class = split_by_class(data_x_train,data_y_train,table)
  print(table)
  print("0 has {0},1 has {1}".format(len(data_by_class[0][1]),len(data_by_class[1][1])))
  train_data = format_data(data_x_train,entries)
  train_labels = nominal_to_int(data_y_train,table)
  eval_data = format_data(data_x_test,entries)
  eval_labels = nominal_to_int(data_y_test,table)
  data_by_class = split_by_class(data_x_test,data_y_test,table)
  eval_data0 = format_data(data_by_class[0][0],entries)
  eval_label0 = nominal_to_int(data_by_class[0][1],table)
  eval_data1 = format_data(data_by_class[1][0],entries)
  eval_label1 = nominal_to_int(data_by_class[1][1],table)

  regr = linear_model.LinearRegression()
  regr.fit(train_data, train_labels)
  print('Coefficients: \n', regr.coef_)
  
  predict_y = regr.predict(eval_data)

  count=0
  for i in range(len(eval_labels)):
    if round(predict_y[i,1])==eval_labels[i,1] and round(predict_y[i,0])==eval_labels[i,0]:
      count+=1
  print("Linear regression accuracy={0}".format(count/len(predict_y)))

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
  print("Optimization Started!")
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

    # Test model
    pred = tf.nn.softmax(logits)  # Apply softmax to logits
    classification = tf.argmax(pred,1)
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(Y, 1))
    # Calculate accuracy
    result=sess.run(classification,feed_dict={X: eval_data1})
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    #print(result)
    # Calculate accuracy
    print(table)
    print("Accuracy:", accuracy.eval({X: eval_data, Y: eval_labels}))
    tf=accuracy.eval({X: eval_data0, Y: eval_label0})
    print("Accuracy label 0:",tf)
    tp=accuracy.eval({X: eval_data1, Y: eval_label1})
    print("Accuracy label 1:", tp)
    print("Precision={0},recall={1}".format(tp,tp*len(eval_data0)/(tp*len(eval_data0)+(1-tf)*len(eval_data1))))

