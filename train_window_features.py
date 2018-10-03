from sklearn import svm
from training_lib import train_max
from training_lib import split_train_test
from training_lib import nominal_to_int
from training_lib import format_data
from training_lib import split_by_class
import csv
import sys
import numpy as np


def read_data(filename):
    data_x=[]
    data_y=[]
    fatals=0
    dates=[]
    fatal_dates=[]
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
            del x["FATAL"]
            del x["LEAD_TIME"]
            del x["DATE"]
            del x["FATAL_START_DATE"]
            del x["LOCATION_PINPOINT"]
            del x["LOCATION_RECOVERY"]
            del x["FATAL_LOCATIONS"]
            for k in x:
                x[k]=float(x[k])
            data_x.append(x)
    print("fatal data size is {0}".format(fatals))
    return data_x,data_y,dates,fatal_dates

def SVM_prediction(train_data, train_labels,eval_data,eval_labels):
	clf = svm.SVC()
	clf.fit(train_data, train_labels)
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
			predict_y[i]=1
			if predict_y[i]==eval_labels[i]:
				tp+=1
			else:
				fn+=1
	print('tp={0},tn={1},fp={2},fn={3},len(eval_labels)={4}'.format(tp,tn,fp,fn,len(eval_labels)))
	print('SVM accuracy={0},precision={1},recall={2},f1={3}'.format((tp+tn)/len(eval_labels),tp/(tp+fp),tp/(tp+fn),2*tp/(2*tp+fn+fp)))

def normalize_features(data):
	medians=[]
	variances=[]
	for i in range(0,data.shape[1]):
		medians.append(np.median(data[:,i]))
		variances.append(np.std(data[:,i]))
	for i in range(0,data.shape[1]):
		if variances[i]>0:
			data[:,i]=np.divide(np.subtract(data[:,i],medians[i]),variances[i])
		else:
			data[:,i]=0

def normalize_features_eval(data,eval_data):
	medians=[]
	variances=[]
	for i in range(0,data.shape[1]):
		medians.append(np.median(data[:,i]))
		variances.append(np.std(data[:,i]))
	for i in range(0,data.shape[1]):
		if variances[i]>0:
			eval_data[:,i]=np.divide(np.subtract(eval_data[:,i],medians[i]),variances[i])
		else:
			eval_data[:,i]=0	

if __name__ == "__main__":
	data_x,data_y,dates,fatal_dates=read_data(sys.argv[1])
	data_x_train,data_y_train,data_x_test,data_y_test,test_dates,fatal_dict=split_train_test(data_x,data_y,dates,fatal_dates,train_max)
	entries=data_x[0].keys()
	print(len(entries))
	table={}
	table[0]=0
	table[1]=1
	train_data = format_data(data_x_train,entries)
	train_labels = nominal_to_int(data_y_train,table)
	eval_data = format_data(data_x_test,entries)
	eval_labels = nominal_to_int(data_y_test,table)
	normalize_features(train_data)
	normalize_features_eval(format_data(data_x,entries),eval_data)
	data_by_class = split_by_class(data_x_test,data_y_test,test_dates,table)

	eval_data0 = format_data(data_by_class[0][0],entries)
	eval_label0 = nominal_to_int(data_by_class[0][1],table)
	eval_dates0 = data_by_class[0][2]
	eval_data1 = format_data(data_by_class[1][0],entries)
	eval_label1 = nominal_to_int(data_by_class[1][1],table)
	eval_dates1 = data_by_class[1][2]
	print("eval 0 has {0}, 1 has {1}".format(eval_label0.shape[0],eval_label1.shape[0]))
	print(train_data.shape,train_labels.shape)
	SVM_prediction(train_data, train_labels,eval_data,eval_labels)
