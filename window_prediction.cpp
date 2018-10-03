#include "amase.h"
#define CLUSTER_LOCATION 1

void window_normalize_features(std::vector<Feature>* result){
	unsigned i,j;
	std::vector<DTYPE>* max=new std::vector<DTYPE>(result[0][0].features->size());
	std::vector<DTYPE>* min=new std::vector<DTYPE>(result[0][0].features->size());
	std::fill(min->begin(),min->end(),-1);
	std::fill(max->begin(),max->end(),-1);
	for(i=0;i<result->size();i++){
		for(j=0;j<result[0][i].features->size();j++){
			if(min[0][j]==-1||min[0][j]>result[0][i].features[0][j]){
				min[0][j]=result[0][i].features[0][j];
			}
			if(max[0][j]==-1||max[0][j]<result[0][i].features[0][j]){
				max[0][j]=result[0][i].features[0][j];
			}
		}
	}
	for(i=0;i<result->size();i++){
		for(j=0;j<result[0][i].features->size();j++){
			result[0][i].features[0][j]=(max[0][j]-result[0][i].features[0][j])/(max[0][j]-min[0][j]);
		}
	}
	delete min;
	delete max;
}

void window_normalize_features2(std::vector<Feature>* result){
	unsigned i,j;
	std::vector<std::vector<DTYPE>*>* median=new std::vector<std::vector<DTYPE>*>(result[0][0].features->size());
	std::vector<DTYPE>* variance=new std::vector<DTYPE>(result[0][0].features->size());
	std::vector<DTYPE>* mean=new std::vector<DTYPE>(result[0][0].features->size());
	std::vector<DTYPE>* medians=new std::vector<DTYPE>(result[0][0].features->size());
	for(j=0;j<result[0][0].features->size();j++){
		median[0][j]=new std::vector<DTYPE>;
		variance[0][j]=0;
		mean[0][j]=0;
	}
	for(i=0;i<result->size();i++){
		for(j=0;j<result[0][i].features->size();j++){
			median[0][j]->push_back(result[0][i].features[0][j]);
			mean[0][j]+=result[0][i].features[0][j];
		}
	}
	for(j=0;j<result[0][0].features->size();j++){
		mean[0][j]/=result->size();
	}
	for(i=0;i<result->size();i++){
		for(j=0;j<result[0][i].features->size();j++){
			variance[0][j]+=(mean[0][j]-result[0][i].features[0][j])*(mean[0][j]-result[0][i].features[0][j]);
		}
	}
	for(j=0;j<result[0][0].features->size();j++){
		variance[0][j]=sqrt(variance[0][j]/result->size());
		std::sort(median[0][j]->begin(),median[0][j]->end());
		medians[0][j]=median[0][j][0][median[0][j]->size()/2];
		printf("feature %u, median=%lf,std=%lf\n",j,medians[0][j],variance[0][j]);
		delete median[0][j];
	}
	for(i=0;i<result->size();i++){
		for(j=0;j<result[0][i].features->size();j++){
			if(variance[0][j]==0){
				result[0][i].features[0][j]=0;
			}else{
				result[0][i].features[0][j]=(result[0][i].features[0][j]-medians[0][j])/variance[0][j];
			}
		}
	}
	delete variance;
	delete medians;
	delete median;
}

std::vector<Feature>* period_based_features(std::vector<std::vector<char*>*>* warn_table,std::vector<std::vector<char*>*>* fatal_table,std::vector<Interval>* fatal_t_clusters,std::vector<DWORD> *warn_dates,std::vector<DWORD> *fatal_dates,DWORD period,DWORD observe_size,DWORD break_down){
	std::vector<Feature>* result=new std::vector<Feature>;
	Feature feature;
	DWORD start,observation_window,current_window,slice,fatal_location_count,warn_location_count;
	unsigned i,warn_index=0,fatal_index=0,warn_index2=0,fatal_index2=0,fatal_index3=0,temp;
	DWORD info_num,warn_num,fatal_num,acm_info_num,acm_warn_num,acm_fatal_num,itv_num2_fatal,qualified_warns,total_warn_count;
	DTYPE info_num_mean,warn_num_mean,fatal_num_mean,info_num_variance,warn_num_variance,fatal_num_variance;
	std::vector<DWORD>* acm_info_nums=new std::vector<DWORD>(break_down);
	std::vector<DWORD>* acm_warn_nums=new std::vector<DWORD>(break_down);
	std::vector<DWORD>* acm_fatal_nums=new std::vector<DWORD>(break_down);
	start=warn_dates[0][0];
	itv_num2_fatal=start;
	slice=period*(observe_size+1)/break_down;
	std::map<std::string,DWORD>* warn_locations=new std::map<std::string,DWORD>;
	std::set<std::string>* fatal_locations=new std::set<std::string>;
	std::set<std::string>::iterator it;
	std::map<std::string,DWORD>::iterator it2;
	char* feature_name;
	while(start<warn_dates[0][warn_dates->size()-1]){
		info_num=0;
		warn_num=0;
		fatal_num=0;
		acm_info_num=0;
		acm_warn_num=0;
		acm_fatal_num=0;
		std::fill(acm_info_nums->begin(),acm_info_nums->end(),0);
		std::fill(acm_warn_nums->begin(),acm_warn_nums->end(),0);
		std::fill(acm_fatal_nums->begin(),acm_fatal_nums->end(),0);
		//Distribution of WARN/INFO features
		observation_window=start;
		warn_index2=warn_index;
		fatal_index2=fatal_index;
		info_num_mean=0;
		fatal_num_mean=0;
		warn_num_mean=0;
		info_num_variance=0;
		fatal_num_variance=0;
		warn_num_variance=0;
		total_warn_count=0;
		for(i=0;i<(unsigned)break_down;i++){
			observation_window+=slice;
			//For each slice, count the number of WARN/INFO events within it.
			while(warn_index2<warn_dates->size()&&warn_dates[0][warn_index2]<=observation_window){
				if(strcmp(warn_table[0][COL_SEVERITY][0][warn_index2],"INFO")==0){
					acm_info_nums[0][i]++;
				}else{
					acm_warn_nums[0][i]++;
				}
				feature_name=extract_location_level(warn_table[0][COL_LOCATION][0][warn_index2],CLUSTER_LOCATION);
				std::string s2(feature_name);
				total_warn_count++;
				if(warn_locations->find(s2)!=warn_locations->end()){
					warn_locations[0][s2]+=1;
				}else{
					warn_locations[0][s2]=1;
				}
				Free(feature_name);
				warn_index2++;
			}
			//For each slice, count the number of FATAL events within it.
			while(fatal_index2<fatal_dates->size()&&fatal_dates[0][fatal_index2]<=observation_window){
				acm_fatal_nums[0][i]++;
				fatal_index2++;
			}
			info_num_mean+=acm_info_nums[0][i];
			warn_num_mean+=acm_warn_nums[0][i];
			fatal_num_mean+=acm_fatal_nums[0][i];
		}
		//Compute mean and variance of the observation window breakdown counts.
		info_num_mean/=break_down;
		warn_num_mean/=break_down;
		fatal_num_mean/=break_down;
		for(i=0;i<(unsigned)break_down;i++){
			info_num_variance+=(acm_info_nums[0][i]-info_num_mean)*(acm_info_nums[0][i]-info_num_mean);
			warn_num_variance+=(acm_warn_nums[0][i]-warn_num_mean)*(acm_warn_nums[0][i]-warn_num_mean);
			fatal_num_variance+=(acm_fatal_nums[0][i]-fatal_num_mean)*(acm_fatal_nums[0][i]-fatal_num_mean);
		}
		info_num_variance/=break_down;
		warn_num_variance/=break_down;
		fatal_num_variance/=break_down;
/*
		if(info_num_variance<0){
			printf("info %lf\n",info_num_variance);
		}
		if(warn_num_variance<0){
			printf("warn %lf\n",warn_num_variance);
		}
		if(fatal_num_variance<0){
			printf("fatal %lf\n",fatal_num_variance);
		}
*/
		//Prepare variables for other features
		observation_window=start+period*observe_size;
		current_window=start+period*(observe_size+1);
		warn_index2=warn_index;
		fatal_index2=fatal_index;
		//WARN/INFO observation window features
		while(warn_index2<warn_dates->size()&&warn_dates[0][warn_index2]<=observation_window){
			if(strcmp(warn_table[0][COL_SEVERITY][0][warn_index2],"INFO")==0){
				acm_info_num++;
			}else{
				acm_warn_num++;				
			}
			warn_index2++;
		}
		//WARN/INFO current window features
		while(warn_index2<warn_dates->size()&&warn_dates[0][warn_index2]<=current_window){
			if(strcmp(warn_table[0][COL_SEVERITY][0][warn_index2],"INFO")==0){
				acm_info_num++;
				info_num++;
			}else{
				acm_warn_num++;
				warn_num++;
			}
			warn_index2++;
		}
		
		//FATAL observation window features
		while(fatal_index2<fatal_dates->size()&&fatal_dates[0][fatal_index2]<=observation_window){
			acm_fatal_num++;
			//Keep track of the last fatal event time stamp
			itv_num2_fatal=fatal_dates[0][fatal_index2];
			fatal_index2++;
		}
		//FATAL current window features
		while(fatal_index2<fatal_dates->size()&&fatal_dates[0][fatal_index2]<=current_window){
			fatal_num++;
			acm_fatal_num++;
			//Keep track of the last fatal event time stamp
			itv_num2_fatal=fatal_dates[0][fatal_index2];
			fatal_index2++;
		}

		//Build features
		feature.features=new std::vector<DTYPE>;
		feature.label=0;
		feature.lead_time=-1;
		feature.start_date=current_window+period;
		//Search for fatal events in the prediction window
		observation_window=start+period*(observe_size+2);
		//Feature class 1
		feature.features->push_back(info_num);
		feature.features->push_back(warn_num);
		feature.features->push_back(fatal_num);
		//Feature class 2
		feature.features->push_back(acm_info_num);
		feature.features->push_back(acm_warn_num);
		feature.features->push_back(acm_fatal_num);
		//Feature class 3
		feature.features->push_back(info_num_mean);
		feature.features->push_back(warn_num_mean);
		feature.features->push_back(fatal_num_mean);
		feature.features->push_back(sqrt(info_num_variance));
		feature.features->push_back(sqrt(warn_num_variance));
		feature.features->push_back(sqrt(fatal_num_variance));
		for(i=0;i<acm_info_nums->size();i++){
			feature.features->push_back(acm_info_nums[0][i]);
		}
		for(i=0;i<acm_warn_nums->size();i++){
			feature.features->push_back(acm_warn_nums[0][i]);
		}
		for(i=0;i<acm_info_nums->size();i++){
			feature.features->push_back(acm_fatal_nums[0][i]);
		}
		//Feature class 4
		feature.features->push_back(current_window-itv_num2_fatal);
		//Add new features
		feature.start_date=observation_window;
		temp=fatal_index3;
		while(fatal_index3<fatal_t_clusters->size()&&fatal_dates[0][fatal_t_clusters[0][fatal_index3].start]<=observation_window){
			feature.label=1;
			feature.fatal_start_date=fatal_dates[0][fatal_t_clusters[0][fatal_index3].start];
			feature.lead_time=fatal_dates[0][fatal_t_clusters[0][fatal_index3].start]-current_window;
			feature.warn_location_size=warn_locations->size();
			feature.fatal_location_size=fatal_locations->size();
			fatal_locations->clear();
			for(i=fatal_t_clusters[0][fatal_index3].start;i<(unsigned)fatal_t_clusters[0][fatal_index3].end;i++){
				feature_name=extract_location_level(fatal_table[0][COL_LOCATION][0][i],CLUSTER_LOCATION);
				std::string s2(feature_name);
				fatal_locations->insert(s2);
				Free(feature_name);
			}
			fatal_location_count=0;
			for(it=fatal_locations->begin();it!=fatal_locations->end();++it){
				if(warn_locations->find(*it)!=warn_locations->end()&&(warn_locations[0][*it]+.0)/total_warn_count>=0){
					fatal_location_count++;
				}
			}
			warn_location_count=0;
			qualified_warns=0;
			for(it2=warn_locations->begin();it2!=warn_locations->end();++it2){
				if((it2->second+.0)/total_warn_count>=0&&fatal_locations->find(it2->first)!=fatal_locations->end()){
					warn_location_count++;
				}
				if((it2->second+.0)/total_warn_count>=0){
					qualified_warns++;
				}
			}
			if(qualified_warns==0){
				feature.location_recovery=-1;
			}else{
				feature.location_recovery=(warn_location_count+.0)/qualified_warns;
			}
			feature.location_pinpoint=(fatal_location_count+.0)/fatal_locations->size();
			result->push_back(feature);
			fatal_index3++;
		}
		if(temp==fatal_index3){
			feature.lead_time=-1;
			feature.label=0;
			feature.location_pinpoint=-1;
			feature.location_recovery=-1;
			feature.fatal_start_date=-1;
			feature.warn_location_size=warn_locations->size();
			feature.fatal_location_size=0;
			result->push_back(feature);
		}
		//slide window to next
		start+=period;
		while(warn_index<warn_dates->size()&&warn_dates[0][warn_index]<start){
			warn_index++;
		}
		while(fatal_index<fatal_dates->size()&&fatal_dates[0][fatal_index]<start){
			fatal_index++;
		}
		warn_locations->clear();
	}
	//window_normalize_features2(result);
	delete warn_locations;
	delete acm_info_nums;
	delete acm_warn_nums;
	delete acm_fatal_nums;
	return result;
}
