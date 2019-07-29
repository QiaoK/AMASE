from sklearn import svm
from training_lib import train_max
from training_lib import split_train_test
from training_lib import nominal_to_int
from training_anomalies import format_data
from training_lib import split_by_class
import csv
import sys
import numpy as np
import random

def RepresentsInt(s):
    try: 
        int(s)
        return True
    except ValueError:
        return False

def read_data(filename):
    data_x=[]
    data_y=[]
    fatals=0
    dates=[]
    fatal_dates=[]
    delete_keys=[]
    with open(filename) as csvfile:
        reader = csv.DictReader(csvfile)
        for x in reader:
            temp=int(x["FATAL"])
            if temp==1:
                fatals+=1
            if temp==-1:
                temp=0
            data_y.append(temp)
            dates.append(int(x["DATE"]))
            fatal_dates.append(int(x["FATAL_START_DATE"]))
            if len(delete_keys)==0:
                for k in x:
                    if not RepresentsInt(k):
                        delete_keys.append(k)
            for k in delete_keys:
                del x[k]
            for k in x:
                x[k]=float(x[k])
            data_x.append(x)
 
    print("fatal data size is {0}".format(fatals))
    return data_x,data_y,dates,fatal_dates

def SVM_prediction(train_data, train_labels,eval_data,eval_labels):
	clf = svm.SVC(gamma='scale')
	clf.fit(train_data, train_labels)
        print('svm fit finished')
	predict_y = clf.predict(eval_data)
	count=0.0
	tp=.0
	fp=.0
	tn=.0
	fn=.0
	for i in range(len(eval_labels)):
		if predict_y[i]<0.5:
			predict_y[i]=0
			if predict_y[i]==eval_labels[i]:
				tn+=1
			else:
				fp+=1
		else:
			if predict_y[i]==eval_labels[i]:
				tp+=1
			else:
				fn+=1
	print('tp={0},tn={1},fp={2},fn={3}'.format(tp,tn,fp,fn))
	print('SVM accuracy={0},precision={1},recall={2}'.format((tp+tn)/len(eval_labels),tp/(tp+fp),tp/(tp+fn)))

def normalize_features(data):
	medians=[]
	variances=[]
	for i in range(0,data.shape[1]):
		medians.append(np.median(data[:,i]))
		variances.append(np.var(data[:,i]))
	for i in range(0,data.shape[1]):
		data[:,i]=np.divide(np.subtract(data[:,i],medians[i]),variances[i])
	
def replicate_data(data_x,data_y,target_class,duplicates):
    class_size=0
    for i in range(0,len(data_y)):
        if data_y[i]==target_class:
            class_size+=1
    copy_x=np.empty((class_size,data_x.shape[1]))
    copy_y=np.empty((class_size))
    index=0
    for i in range(0,data_x.shape[0]):
        if data_y[i]==target_class:
            copy_x[index]=data_x[i]
            copy_y[index]=data_y[i]
            index+=1
    result_x=np.empty((data_x.shape[0]+(duplicates-1)*class_size,data_x.shape[1]))
    result_y=np.empty((data_y.shape[0]+(duplicates-1)*class_size))
    result_x[0:data_x.shape[0]]=data_x[0:data_x.shape[0]]
    result_y[0:data_y.shape[0]]=data_y[0:data_y.shape[0]]
    for i in range(0,duplicates-1):
        result_x[(data_x.shape[0]+i*class_size):(data_x.shape[0]+(i+1)*class_size)]=copy_x[0:class_size]
        result_y[(data_y.shape[0]+i*class_size):(data_y.shape[0]+(i+1)*class_size)]=copy_y[0:class_size]
    return result_x,result_y

def shuffle_data(data_x,data_y):
    index=[i for i in range(data_x.shape[0])]
    random.seed(12345)
    random.shuffle(index)
    result_x=np.empty(data_x.shape)
    result_y=np.empty(data_y.shape)
    for i in range(0,len(index)):
        result_x[i]=data_x[index[i]]
        result_y[i]=data_y[index[i]]
    return result_x,result_y


if __name__ == "__main__":
	data_x,data_y,dates,fatal_dates=read_data(sys.argv[1])
	data_x_train,data_y_train,data_x_test,data_y_test,test_dates,fatal_dict=split_train_test(data_x,data_y,dates,fatal_dates,train_max)
	entries=data_x[0].keys()
	print(len(entries))
        print(entries)
	table={}
	table[0]=0
	table[1]=1
	train_data = format_data(data_x_train,entries)
	train_labels = nominal_to_int(data_y_train,table)
	eval_data = format_data(data_x_test,entries)
	eval_labels = nominal_to_int(data_y_test,table)
	normalize_features(train_data)
	normalize_features(eval_data)

	print(train_data.shape,train_labels.shape)
        #train_data,train_labels = replicate_data(train_data,train_labels,1,2)
        train_data,train_labels=shuffle_data(train_data,train_labels)

	print(train_data.shape,train_labels.shape)
	SVM_prediction(train_data, train_labels,eval_data,eval_labels)
