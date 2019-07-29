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

# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Hyper parameters
output_size = 2
learning_rate = 0.001
batch_size = 1024
#lead_time_interval = 36000
#lead_time_stride = 1

# Convolutional neural network (two convolutional layers)

class ConvNet(nn.Module):
    def __init__(self,feature_length,sequence_length,num_classes=output_size):
        super(ConvNet, self).__init__()
        filter1=1
        hidden=512
        hidden2=256
        kernel_size=15
        self.layer1 = nn.Sequential(
            nn.Conv1d(1, filter1, kernel_size=[kernel_size], stride=1, padding=[int(kernel_size/2)]),
            nn.BatchNorm1d(filter1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=[kernel_size], stride=[kernel_size]))
        self.layer2 = nn.Sequential(
            nn.Conv1d(filter1, filter1, kernel_size=[3], stride=1, padding=[1]),
            nn.BatchNorm1d(filter1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=[2], stride=[2]))
        self.fc_hidden = nn.Linear(int(int(sequence_length/kernel_size)/2)*filter1+feature_length, hidden)
        #self.fc_hidden = nn.Linear(feature_length, hidden)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.21)
        self.fc_hidden2 = nn.Linear(hidden, hidden2)
        self.fc_hidden3 = nn.Linear(hidden2, hidden2)
        self.fc = nn.Linear(hidden2, num_classes)
        self.output = nn.Softmax()

    def forward(self, x, y):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = torch.cat((out,y),1)
        out = self.fc_hidden(out)
        out = self.dropout(out)
        out = self.relu(out)
        out = self.fc_hidden2(out)
        out = self.dropout(out)
        out = self.relu(out)
        out = self.fc(out)
        out = self.output(out)
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
    counter_dict={}
    total=0
    for i in range(0,len(data)):
        if data[i] not in counter_dict:
            counter_dict[data[i]]=1
        else:
            counter_dict[data[i]]+=1
    for k in counter_dict:
        total+=counter_dict[k]
    print(counter_dict)
    return total-counter_dict[0]

'''
Read data and build features.
'''
def read_data(filename,lead_time_interval,lead_time_stride):
    dates=[]
    fatal_index=[]
    fatal_time_span=[]
    lead_times=[]
    mean_lead_time={}
    conv_features = None
    counter=0
    with open(filename) as csvfile:
        reader = csv.DictReader(csvfile)
        rows = sum(1 for row in reader)
    print("total number of rows = {0}".format(rows))
    with open(filename) as csvfile:
        reader = csv.DictReader(csvfile)
        entries = list(reader.fieldnames)
	remove_entries = []
	for k in entries:
		if 'NOT_FEATURE' in k:
			remove_entries.append(k)
	for k in remove_entries:
		entries.remove(k)
        entries.remove("FATAL_INDEX_NOT_FEATURE")
        entries.remove("LEAD_TIME_1_NOT_FEATURE")
        entries.remove("DATE_NOT_FEATURE")
        entries.remove("LOCATION_RECOVERY_1_NOT_FEATURE")
        entries.remove("LOCATION_RECOVERY_2_NOT_FEATURE")
        entries.remove("LOCATION_RECOVERY_3_NOT_FEATURE")
        entries.remove("LOCATION_RECOVERY_4_NOT_FEATURE")
        feature_names=[]
        conv_names=[]
        for k in entries:
            if 'TEMPORAL_INTERVAL' in k or 'SPATIAL_INTERVAL' in k:
                conv_names.append(k)
            else:
                feature_names.append(k)
        conv_names.sort()
        feature_names.sort()
        features=np.empty((rows,len(feature_names)))
        data_y=np.empty((rows))
        if len(conv_names)>0:
            conv_features=np.empty((rows,len(conv_names)))
        for x in reader:
            date=int(x["DATE_NOT_FEATURE"])
            lead_time=int(x["LEAD_TIME_1_NOT_FEATURE"])
            temp=int(lead_time/lead_time_interval)
            if temp>=lead_time_stride:
                temp=0
            else:
                temp+=1
            if temp not in mean_lead_time:
                mean_lead_time[temp]=[0.0,1]
                mean_lead_time[temp][0]=lead_time
            else:
                mean_lead_time[temp][0]+=lead_time
                mean_lead_time[temp][1]+=1
            fatal_event=int(x["FATAL_INDEX_NOT_FEATURE"])
            dates.append(date)
            for i in range(0,len(feature_names)):
                features[counter,i] = float(x[feature_names[i]])
            if len(conv_names) > 0:
                for i in range(0,len(conv_names)):
                    conv_features[counter,i] = float(x[conv_names[i]])
            data_y[counter] = temp
            counter += 1
            fatal_index.append(fatal_event)
            lead_times.append(lead_time)
            if(counter%10000==0):
                print('reading for {0} finished'.format(counter))
    return features,conv_features,data_y,dates,fatal_index,lead_times
'''
    for temp in mean_lead_time:
        mean_lead_time[temp][0]/=mean_lead_time[temp][1]
    print(mean_lead_time)
'''

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def evaluate_specific_fatal(eval_data,eval_labels,fatal_index,fatal_event,model):
    model.eval()
    matched_row=[i for i in range(0,fatal_index.shape[0]) if fatal_index[i]==fatal_event]
    data_x=np.empty(())

def determine_ROC(predicted,eval_labels,positive_class):
    fpr, tpr, thresholds = metrics.roc_curve(eval_labels, 1-predicted[:,0], pos_label=1)
    fp = np.multiply(fpr,eval_labels.shape[0]-positive_class)
    tp = np.multiply(tpr,positive_class)
    fn = np.subtract(np.full(tp.shape,positive_class),tp)
    f1_score = np.divide(tp,np.add(np.multiply(tp,2),np.add(fn,fp)))
    return thresholds[np.argmax(f1_score)]

def evaluate_model(eval_data,eval_conv,eval_labels,model,lead_time_interval,lead_time_stride):
    model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
    batch_size = 4096
    batches = int((eval_labels.shape[0]+batch_size-1)/batch_size)
    total_features=eval_labels.shape[0]
    result=np.empty((eval_labels.shape[0],lead_time_stride+1))
    for i in range(0,batches):
        if (i+1)*batch_size<total_features:
            start=i*batch_size
            end=(i+1)*batch_size
        else:
            start=i*batch_size
            end=total_features
        data_x=eval_data[start:end]
        conv_x=eval_conv[start:end]

        data_x = np.reshape(data_x,(end-start,eval_data.shape[1]))
        data_x = torch.from_numpy(data_x).float()
        data_x = data_x.to(device)

        conv_x = np.reshape(conv_x,(end-start,1,conv_x.shape[1]))
        conv_x = torch.from_numpy(conv_x).float()
        conv_x = conv_x.to(device)        

        outputs = model(conv_x,data_x)
        predicted = outputs.data.cpu().numpy()
        result[start:end,:]=predicted
    return result

def write_all_predictions(fatal_predictions):
    for k in fatal_predictions:
        if k==-1:
            continue
        f=open('fatal_plots/{0}.csv'.format(k),'w')
        fatal=fatal_predictions[k]
        for i in range(0,fatal.shape[0]):
            for j in range(0,fatal.shape[1]):
                if j!=fatal.shape[1]-1:
                    f.write(str(fatal[i,j])+',')
                else:
                    f.write(str(fatal[i,j])+'\n')
        f.close()

def analyze_model(data_y,dates,fatal_index,lead_times,predicted,threshold,lead_time_interval,lead_time_stride):
    result=''
    test_tn=0
    test_fp=0
    test_tp=0
    test_fn=0
    train_tn=0
    train_fp=0
    train_tp=0
    train_fn=0
    fatal_accuracy=0
    fatal_total=0
    fatal_predictions={}
    for i in range(0,len(fatal_index)):
        if dates[i]>train_max:
            temp=np.empty((1,lead_time_stride+3))
            temp[0,0:(lead_time_stride+1)]=predicted[i,:]
            temp[0,lead_time_stride+1]=data_y[i]
            temp[0,lead_time_stride+2]=lead_times[i]

            if fatal_index[i] not in fatal_predictions:
                fatal_predictions[fatal_index[i]]=temp
            else:
                fatal_predictions[fatal_index[i]]=np.append(fatal_predictions[fatal_index[i]],temp,axis=0)
            if data_y[i]==0:
                if predicted[i,0]>threshold:
                    test_tn+=1
                else:
                    test_fp+=1
            else:
                current=predicted[i,1:(lead_time_stride+1)]
                if data_y[i]==np.argmax(current)+1:
                    fatal_accuracy+=1
                fatal_total+=1
                if predicted[i,0]>threshold:
                    test_fn+=1
                else:
                    test_tp+=1
        else:
            if data_y[i]==0:
                if predicted[i,0]>threshold:
                    train_tn+=1
                else:
                    train_fp+=1
            else:
                if predicted[i,0]>threshold:
                    train_fn+=1
                else:
                    train_tp+=1
    train_accuracy=(train_tp+train_tn+.0)/(train_tp+train_tn+train_fp+train_fn)
    test_accuracy=(test_tp+test_tn+.0)/(test_tp+test_tn+test_fp+test_fn)
    train_precision=(train_tp+1.0)/(train_tp+train_fp+1.0)
    test_precision=(test_tp+1.0)/(test_tp+test_fp+1.0)
    train_recall=(train_tp+1.0)/(train_tp+train_fn+1.0)
    test_recall=(test_tp+1.0)/(test_tp+test_fn+1.0)
    fatal_accuracy=(fatal_accuracy+.0)/fatal_total
    result+='Train accuracy={0}, precision={1}, recall={2}\n'.format(train_accuracy,train_precision,train_recall)
    result+='Test accuracy={0}, precision={1}, recall={2},test fatal accuracy={3}\n'.format(test_accuracy,test_precision,test_recall,fatal_accuracy)
    fatal_recovery_rate=0
    for fatal_event in fatal_predictions:
        ground_truth=fatal_predictions[fatal_event][:,lead_time_stride+1]
        safe_prob=fatal_predictions[fatal_event][:,0]
        counter=0
        for i in range(0,ground_truth.shape[0]):
            if ground_truth[i]>0 and safe_prob[i]<threshold:
                counter+=1
        if (counter+.0)/ground_truth.shape[0]>0:
            fatal_recovery_rate+=1
    result+='recovered fatal number={0}, total test fatal={1}\n'.format(fatal_recovery_rate,len(fatal_predictions))
    return fatal_predictions,result


def split_data(data_x,conv_x,data_y,dates,train_max):
  dates=np.asarray(dates)
  split_date=dates.shape[0]
  for i in range(0,dates.shape[0]):
    if dates[i]>train_max:
      split_date=i
      break
  print("train:test={0}:{1},{2}".format(split_date,dates.shape[0]-split_date,(0.0+split_date)/dates.shape[0]))
  return data_x[0:split_date],conv_x[0:split_date],data_y[0:split_date],data_x[split_date:data_x.shape[0]],conv_x[split_date:data_x.shape[0]],data_y[split_date:data_y.shape[0]],dates[split_date:dates.shape[0]]

def convert_to_hdf5_file(filename,data_x,data_y,dates,fatal_index,lead_times,conv_x):
    h = h5py.File(filename,'w')
    h.create_dataset('data_x',data=data_x,dtype=np.float)
    h.create_dataset('data_y',data=data_y,dtype=np.int)
    h.create_dataset('dates',data=dates,dtype=np.int)
    h.create_dataset('fatal_index',data=fatal_index,dtype=np.int)
    h.create_dataset('lead_times',data=lead_times,dtype=np.int)
    if not conv_x is None:
        h.create_dataset('conv_x',data=conv_x,dtype=np.int)
    h.close()

def open_hdf5_file(filename):
    h = h5py.File(filename,'r')
    data_x=h['data_x'][:]
    data_y=h['data_y'][:]
    dates=h['dates'][:]
    fatal_index=h['fatal_index'][:]
    lead_times=h['lead_times'][:]
    conv_x=None
    if 'conv_x' in h:
        conv_x=h['conv_x'][:] 
    h.close()
    return data_x,data_y,dates,fatal_index,lead_times,conv_x

def normalize_data(train_data_x,test_data_x):
    max_x=np.empty(train_data_x.shape[1])
    min_x=np.empty(train_data_x.shape[1])
    for k in range(0,train_data_x.shape[1]):
        max_x[k]=np.max(train_data_x[:,k])
        min_x[k]=np.min(train_data_x[:,k])
        if max_x[k]==min_x[k]:
            train_data_x[:,k]=train_data_x[:,k]*0
        else:
            train_data_x[:,k]=(train_data_x[:,k]-min_x[k]+.0)/(max_x[k]-min_x[k])

    for k in range(0,test_data_x.shape[1]):
        if max_x[k]==min_x[k]:
            test_data_x[:,k]=test_data_x[:,k]*0
        else:
            test_data_x[:,k]=np.minimum(np.maximum((test_data_x[:,k]-min_x[k]+.0)/(max_x[k]-min_x[k]),0),1)

def replicate_data(data_x,conv_x,data_y,target_class,duplicates):
    class_size=0
    for i in range(0,len(data_y)):
        if data_y[i]==target_class:
            class_size+=1
    copy_x=np.empty((class_size,data_x.shape[1]))
    copy_conv_x=np.empty((class_size,conv_x.shape[1]))
    copy_y=np.empty((class_size))
    index=0
    for i in range(0,data_x.shape[0]):
        if data_y[i]==target_class:
            copy_x[index]=data_x[i]
            copy_conv_x[index]=conv_x[i]
            copy_y[index]=data_y[i]
            index+=1
    result_x=np.empty((data_x.shape[0]+(duplicates-1)*class_size,data_x.shape[1]))
    result_conv_x=np.empty((conv_x.shape[0]+(duplicates-1)*class_size,conv_x.shape[1]))
    result_y=np.empty((data_y.shape[0]+(duplicates-1)*class_size))
    result_x[0:data_x.shape[0]]=data_x[0:data_x.shape[0]]
    result_conv_x[0:conv_x.shape[0]]=conv_x[0:conv_x.shape[0]]
    result_y[0:data_y.shape[0]]=data_y[0:data_y.shape[0]]
    for i in range(0,duplicates-1):
        result_x[(data_x.shape[0]+i*class_size):(data_x.shape[0]+(i+1)*class_size)]=copy_x[0:class_size]
        result_conv_x[(conv_x.shape[0]+i*class_size):(conv_x.shape[0]+(i+1)*class_size)]=copy_conv_x[0:class_size]
        result_y[(data_y.shape[0]+i*class_size):(data_y.shape[0]+(i+1)*class_size)]=copy_y[0:class_size]
    return result_x,result_conv_x,result_y

def shuffle_data(data_x,conv_x,data_y):
    index=[i for i in range(data_x.shape[0])]
    random.seed(12345)
    random.shuffle(index)
    result_x=np.empty(data_x.shape)
    result_conv_x=np.empty(conv_x.shape)
    result_y=np.empty(data_y.shape)
    for i in range(0,len(index)):
        result_x[i]=data_x[index[i]]
        result_conv_x[i]=conv_x[index[i]]
        result_y[i]=data_y[index[i]]
    return result_x,result_conv_x,result_y

if __name__ == "__main__":
    if len(sys.argv) != 7:
        print("Usage: python training_lead_time.py feature_file_name training_epochs retrain(0 or 1) reload_data(0 or 1) lead_time_interval lead_time_stride")
        sys.exit()
    # Load training and eval data
    random.seed(5555)
    torch.manual_seed(5555)
    torch.cuda.manual_seed_all(5555)
    training_epochs=int(sys.argv[2])
    retrain=int(sys.argv[3])
    lead_time_interval=int(sys.argv[5])
    lead_time_stride=int(sys.argv[6])

    filename='{0}_{1}_{2}.hdf5'.format(sys.argv[1],lead_time_interval,lead_time_stride)
    if int(sys.argv[4])==1:
      print('start to construct hdf5 file frome csv.')
      data_x,conv_x,data_y,dates,fatal_index,lead_times=read_data(sys.argv[1],lead_time_interval,lead_time_stride)
      convert_to_hdf5_file(filename,data_x,data_y,dates,fatal_index,lead_times,conv_x)
    else:
      print('reusing existing hdf5 file (should check if it is the latest)')
      data_x,data_y,dates,fatal_index,lead_times,conv_x=open_hdf5_file(filename)
    print(data_x.shape,data_y.shape,len(dates))
    train_data,train_conv,train_labels,test_data,test_conv,test_labels,test_dates=split_data(data_x,conv_x,data_y,dates,train_max)
    normalize_data(train_data,test_data)
    
    train_size=train_data.shape[0]
    positive_train_class=summarize_class(train_labels)
    train_data,train_conv,train_labels=replicate_data(train_data,train_conv,train_labels,1,int((train_size+.0+4*positive_train_class-1)/(4*positive_train_class)))
    #train_data,train_conv,train_labels=replicate_data(train_data,train_conv,train_labels,2,12)
    print('augmented training size from {0} to {1}'.format(train_size,train_data.shape[0]))
    train_data,train_conv,train_labels=shuffle_data(train_data,train_conv,train_labels)
    positive_class=summarize_class(data_y)
    summarize_class(train_labels)
    summarize_class(test_labels)

    print(train_data.shape,train_conv.shape,train_labels.shape)
    total_features=train_data.shape[0]
    model = ConvNet(feature_length=train_data.shape[1],sequence_length=train_conv.shape[1],num_classes=lead_time_stride+1).to(device)
    result=open('{0}_{1}_{2}.txt'.format(sys.argv[1],lead_time_interval,lead_time_stride),'w')
    print('start optimization')
    if retrain==1:
        model.train()
        #weights = torch.from_numpy(np.array([.1,.6,0.3])).float().to(device)
        #weights = torch.from_numpy(np.array([.1,.15,.15,.15,.15,.15,.15])).float().to(device)
        #weights = torch.from_numpy(np.array([0.1,.3,.3,.3])).float().to(device)
        #weights = torch.from_numpy(np.array([.6,.4])).float().to(device)
        #criterion = nn.CrossEntropyLoss(weight=weights)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        #optimizer = torch.optim.SGD(model.parameters(), lr = 0.011, momentum=0.66)
        batchs = int((total_features+batch_size-1)/batch_size)
        for epoch in range(training_epochs):
            for i in range(0,batchs):
                if (i+1)*batch_size<total_features:
                    start=i*batch_size
                    end=(i+1)*batch_size
                else:
                    start=i*batch_size
                    end=total_features
                batch_x = train_data[start:end]
                batch_conv_x = train_conv[start:end]
                batch_y = train_labels[start:end]
                #print(batch_x.shape,batch_conv_x.shape,batch_y.shape)
                batch_conv_x = np.reshape(batch_conv_x,(batch_conv_x.shape[0],1,batch_conv_x.shape[1]))
                batch_x = torch.from_numpy(batch_x).float()
                batch_conv_x = torch.from_numpy(batch_conv_x).float()
                batch_y = torch.from_numpy(batch_y).long()
                batch_x = batch_x.to(device)
                batch_conv_x = batch_conv_x.to(device)
                labels = batch_y.to(device)
                # Forward pass
                #print(batch_conv_x.shape)
                #print(batch_x.shape)
                outputs = model(batch_conv_x,batch_x)
                loss = criterion(outputs, labels)
                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            print ('Epoch [{0}/{1}], Loss: {2:.4f}'.format(epoch+1, training_epochs, loss.item()))
            predicted=evaluate_model(data_x,conv_x,data_y,model,lead_time_interval,lead_time_stride)
            threshold=determine_ROC(predicted[0:train_size],data_y[0:train_size],positive_class)
            print('ROC threshold='+str(threshold))
            fatal_predictions,messages=analyze_model(data_y,dates,fatal_index,lead_times,predicted,.5,lead_time_interval,lead_time_stride)
            print(messages)
            result.write(messages+'\n')
            model.train()
        torch.save(model.state_dict(), '{0}_model.ckpt'.format(sys.argv[1]))
        write_all_predictions(fatal_predictions)
    else:
        model.load_state_dict(torch.load('{0}_model.ckpt'.format(sys.argv[1])))
        #messages=evaluate_model(train_data,train_labels,eval_data0,eval_label0,eval_dates0,eval_data1,eval_label1,eval_dates1,fatal_dict,model)
        #print(messages)
        #result.write(messages+'\n')  
    result.close()
