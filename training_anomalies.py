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

# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Hyper parameters
output_size = 2
learning_rate = 0.001
batch_size = 256

# Convolutional neural network (two convolutional layers)

class ConvNet(nn.Module):
    def __init__(self,feature_length, sequence_length, num_classes=output_size):
        super(ConvNet, self).__init__()
        filter=32
        hidden=2048
        kernel_size=15
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, filter, kernel_size=[2,kernel_size], stride=1, padding=[0,int(kernel_size/2)]),
            nn.BatchNorm2d(filter),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=[1,kernel_size], stride=[1,kernel_size]))
        self.layer2 = nn.Sequential(
            nn.Conv2d(filter, filter, kernel_size=[1,3], stride=1, padding=[0,1]),
            nn.BatchNorm2d(filter),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=[1,2], stride=[1,2]))
        self.layer3 = nn.Sequential(
            nn.Conv2d(filter, filter, kernel_size=[1,3], stride=1, padding=[0,1]),
            nn.BatchNorm2d(filter),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=[1,2], stride=[1,2]))
        self.fc_hidden = nn.Linear(int(int(int(sequence_length/kernel_size)/2)/2)*filter+feature_length, hidden)
        self.relu = nn.ReLU()
        self.fc_hidden2 = nn.Linear(hidden, hidden)
        self.fc = nn.Linear(hidden, num_classes)
        #self.output = nn.Softmax()

    def forward(self, x, y):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.reshape(out.size(0), -1)
        out = torch.cat((out,y),1)
        out = self.fc_hidden(out)
        out = self.relu(out)
        out = self.fc_hidden2(out)
        out = self.relu(out)
        out = self.fc(out)
        #out = self.output(out)
        return out

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



def summarize_class(data):
    tmp0=0
    tmp1=0
    for i in range(0,len(data)):
        if data[i]==0:
            tmp0+=1
        elif data[i]==1:
            tmp1+=1
    print("train data class 0 ={0}, class 1 ={1}".format(tmp0,tmp1))

def read_data(filename):
    data_x=[]
    data_y=[]
    #cluster_membership=clustering(sys.argv[2],output_size)
    #print("Cluster number",len(set(cluster_membership)))
    fatals=0
    dates=[]
    fatal_dates=[]
    with open(filename) as csvfile:
        reader = csv.DictReader(csvfile)
        for x in reader:
            #del x[""]
            #del x["START"]
            #del x["END"]
            temp=int(x["FATAL"])
            if temp==1:
                fatals+=1
            if temp==-1:
                temp=0
            #if temp>0:
            #  temp=cluster_membership[temp-1]
            data_y.append(temp)
            dates.append(int(x["DATE"]))
            fatal_dates.append(int(x["FATAL_START_DATE"]))
            del x["FATAL"]
            del x["LEAD_TIME"]
            del x["DATE"]
            del x["FATAL_START_DATE"]
            del x["LOCATION_PINPOINT"]
            del x["LOCATION_RECOVERY"]
            for k in x:
                x[k]=float(x[k])
            data_x.append(x)
    print("fatal data size is {0}".format(fatals))
    return data_x,data_y,dates,fatal_dates

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def determin_ROC_threshold(eval_data,eval_labels,model):
    model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
    scores=np.empty(eval_labels.shape[0])
    positive_class=0
    batch_size = 1024
    batches = (eval_labels.shape[0]+batch_size-1)/batch_size
    total_features=eval_labels.shape[0]
    for i in range(0,batches):
        if (i+1)*batch_size<total_features:
            start=i*batch_size
            end=(i+1)*batch_size
        else:
            start=i*batch_size
            end=total_features
        image=eval_data[1][start:end]
        data_x=eval_data[0][start:end]
        
        images = np.reshape(image,(end-start,1,eval_data[1].shape[1],eval_data[1].shape[2]))
        data_x = np.reshape(data_x,(end-start,eval_data[0].shape[1]))
        images = torch.from_numpy(images).float()
        data_x = torch.from_numpy(data_x).float()
        images = images.to(device)
        data_x = data_x.to(device)
        
        outputs = model(images,data_x)
        predicted = outputs.data.cpu().numpy()
        
 
        for j in range(0,predicted.shape[0]):
            probability=softmax(predicted[j])
            scores[i*batch_size+j]=(10000*probability[1])/(10000*probability[1]+10000*probability[0])
            if eval_labels[i*batch_size+j]==1:
                positive_class+=1
    
    fpr, tpr, thresholds = metrics.roc_curve(eval_labels, scores, pos_label=1)
    fp = np.multiply(fpr,eval_labels.shape[0]-positive_class)
    tp = np.multiply(tpr,positive_class)
    fn = np.subtract(np.full(tp.shape,positive_class),tp)
    f1_score = np.divide(tp,np.add(np.multiply(tp,2),np.add(fn,fp)))
    return thresholds[np.argmax(f1_score)]

def test_model(eval_data,eval_labels,eval_dates,fatal_dict,undetected_fatal,model,threshold=0.5):
    model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
    correct=0
    batch_size = 1024
    batches = (eval_labels.shape[0]+batch_size-1)/batch_size
    total_features=eval_labels.shape[0]
    for i in range(0,batches):
        if (i+1)*batch_size<total_features:
            start=i*batch_size
            end=(i+1)*batch_size
        else:
            start=i*batch_size
            end=total_features
        image=eval_data[1][start:end]
        data_x=eval_data[0][start:end]

        images = np.reshape(image,(end-start,1,eval_data[1].shape[1],eval_data[1].shape[2]))
        data_x = np.reshape(data_x,(end-start,eval_data[0].shape[1]))
        images = torch.from_numpy(images).float()
        data_x = torch.from_numpy(data_x).float()
        images = images.to(device)
        data_x = data_x.to(device)
        
        outputs = model(images,data_x)
        predicted = outputs.data.cpu().numpy()
        for j in range(0,predicted.shape[0]):
            probability=softmax(predicted[j])
            if (10000*probability[1])/(10000*probability[1]+10000*probability[0])<threshold:
                prediction=0
            else:
                prediction=1
                if eval_dates[i*batch_size+j] in fatal_dict:
                    for x in fatal_dict[eval_dates[i*batch_size+j]]:
                        if x in undetected_fatal:
                            undetected_fatal.remove(x)
            if prediction==int(eval_labels[i*batch_size+j]):
                correct+=1
    return correct

def evaluate_model(train_data,train_labels,eval_data0,eval_label0,eval_dates0,eval_data1,eval_label1,eval_dates1,fatal_dict,model):
    threshold=determin_ROC_threshold(train_data,train_labels,model)
    undetected_fatal=set()
    for v in fatal_dict.values():
        for x in v:
            undetected_fatal.add(x)
    total_fatals=len(undetected_fatal)
    tn=test_model(eval_data0,eval_label0,eval_dates0,fatal_dict,undetected_fatal,model,threshold)
    tp=test_model(eval_data1,eval_label1,eval_dates1,fatal_dict,undetected_fatal,model,threshold)
    fp=eval_label0.shape[0]-tn
    fn=eval_label1.shape[0]-tp
    f1_score=2.0*tp/(2*tp+fn+fp)
    messages='test accuracy={0},precision={1},recall={2},fatal recovered={3}/{4}={5},threshold={6},f1={7},tp={8},tf={9},fn={10}'.format((.0+tp+tn)/(tp+tn+fn+fp),(.0+tp)/(tp+fp),(.0+tp)/(tp+fn),total_fatals-len(undetected_fatal),total_fatals,(total_fatals-len(undetected_fatal)+.0)/total_fatals,threshold,f1_score,tp,fp,fn)
    return messages

def normalize_data(train_data_x,test_data_x):
    entries=train_data_x[0].keys()
    max_x={}
    min_x={}
    for k in entries:
        max_x[k]=train_data_x[0][k]
        min_x[k]=train_data_x[0][k]
    for i in range(1,len(train_data_x)):
        for k in entries:
            if max_x[k]<train_data_x[i][k]:
                max_x[k]=train_data_x[i][k]
            if min_x[k]>train_data_x[i][k]:
                min_x[k]=train_data_x[i][k]

    for k in entries:
        if max_x[k]==min_x[k]:
            print(k,max_x[k],min_x[k])

    for i in range(0,len(train_data_x)):
        for k in entries:
            if max_x[k]==min_x[k]:
                train_data_x[i][k]=0
            else:
                train_data_x[i][k]=(train_data_x[i][k]-min_x[k]+.0)/(max_x[k]-min_x[k])

    for i in range(0,len(test_data_x)):
        for k in entries:
            if max_x[k]==min_x[k]:
                test_data_x[i][k]=0
            else:
                test_data_x[i][k]=(test_data_x[i][k]-min_x[k]+.0)/(max_x[k]-min_x[k])
                if test_data_x[i][k]>1:
                    test_data_x[i][k]=1.0
                elif test_data_x[i][k]<0:
                    test_data_x[i][k]=0.0

if __name__ == "__main__":
    # Load training and eval data
    random.seed(5555)
    torch.manual_seed(5555)
    torch.cuda.manual_seed_all(5555)
    training_epochs=int(sys.argv[2])
    retrain=int(sys.argv[3])
    data_x,data_y,dates,fatal_dates=read_data(sys.argv[1])
    entries=data_x[0].keys()
    #table=create_nominal_table(data_y)
    table={}
    table[0]=0
    table[1]=1
    print(table)
    data_x_train,data_y_train,data_x_test,data_y_test,test_dates,fatal_dict=split_train_test(data_x,data_y,dates,fatal_dates,train_max)
    
    #normalize_data(data_x_train,data_x_test)
    '''data_x_train=data_x[0:int(len(data_x)*0.8)]
    data_y_train=data_y[0:int(len(data_x)*0.8)]
    data_x_test=data_x[int(len(data_x)*0.8+1):len(data_x)]
    data_y_test=data_y[int(len(data_y)*0.8+1):len(data_y)]'''
    #print("last date=",dates[int(len(data_x)*0.8)])
    summarize_class(data_y_train)
    replicate_class(table,data_x_train,data_y_train,.6,1,len(data_y_train))
    print("data augmentation")
    summarize_class(data_y_train)
    #data_by_class = split_by_class(data_x_train,data_y_train,table)
    train_data = format_data(data_x_train,entries)
    train_labels = nominal_to_int(data_y_train,table)
    eval_data = format_data(data_x_test,entries)
    eval_labels = nominal_to_int(data_y_test,table)
    data_by_class = split_by_class(data_x_test,data_y_test,test_dates,table)

    eval_data0 = format_data(data_by_class[0][0],entries)
    eval_label0 = nominal_to_int(data_by_class[0][1],table)
    eval_dates0 = data_by_class[0][2]
    eval_data1 = format_data(data_by_class[1][0],entries)
    eval_label1 = nominal_to_int(data_by_class[1][1],table)
    eval_dates1 = data_by_class[1][2]
    print("eval 0 has {0}, 1 has {1}".format(eval_label0.shape[0],eval_label1.shape[0]))
    print(train_data[0].shape,train_labels.shape)
    
    regr = linear_model.LinearRegression()
    regr.fit(train_data[0], train_labels)
    predict_y = regr.predict(eval_data[0])
    #print('Coefficients: \n', regr.coef_)
    count=0.0
    for i in range(len(eval_labels)):
        if predict_y[i]<0.5:
            predict_y[i]=0
        else:
            predict_y[i]=1
        if predict_y[i]==eval_labels[i]:
            count+=1
    print('linear regression accuracy={0}'.format(count/len(predict_y)))
    
    total_features=train_data[0].shape[0]
    print('total_features={0}, feature attribute length={1}, convolutional length={2}'.format(total_features,train_data[0].shape[1],train_data[1].shape[2]))
    model = ConvNet(feature_length=train_data[0].shape[1],sequence_length=train_data[1].shape[2],num_classes=output_size).to(device)
    result=open('{0}.txt'.format(sys.argv[1]),'w')
    if retrain==1:
        model.train()
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        batchs = (total_features+batch_size-1)/batch_size
        for epoch in range(training_epochs):
            
            for i in range(0,batchs):
                if (i+1)*batch_size<total_features:
                    start=i*batch_size
                    end=(i+1)*batch_size
                else:
                    start=i*batch_size
                    end=total_features
                batch_x = train_data[0][start:end]
                batch_conv_x = train_data[1][start:end]
                batch_y = train_labels[start:end]
                #print(batch_x.shape,batch_conv_x.shape,batch_y.shape)
                batch_conv_x = np.reshape(batch_conv_x,(batch_conv_x.shape[0],1,batch_conv_x.shape[1],batch_conv_x.shape[2]))
                batch_x = torch.from_numpy(batch_x).float()
                batch_conv_x = torch.from_numpy(batch_conv_x).float()
                batch_y = torch.from_numpy(batch_y).long()
                batch_x = batch_x.to(device)
                batch_conv_x = batch_conv_x.to(device)
                labels = batch_y.to(device)
                # Forward pass
                outputs = model(batch_conv_x,batch_x)
                loss = criterion(outputs, labels)
                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            print ('Epoch [{0}/{1}], Loss: {2:.4f}'.format(epoch+1, training_epochs, loss.item()))
            messages=evaluate_model(train_data,train_labels,eval_data0,eval_label0,eval_dates0,eval_data1,eval_label1,eval_dates1,fatal_dict,model)
            print(messages)
            result.write(messages+'\n')
            model.train()
        # Save the model checkpoint
        torch.save(model.state_dict(), '{0}_model.ckpt'.format(sys.argv[1]))
    else:
        model.load_state_dict(torch.load('{0}_model.ckpt'.format(sys.argv[1])))
        messages=evaluate_model(train_data,train_labels,eval_data0,eval_label0,eval_dates0,eval_data1,eval_label1,eval_dates1,fatal_dict,model)
        print(messages)
        result.write(messages+'\n')
    result.close()
