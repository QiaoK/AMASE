#include "amase.h"
#define CLUSTER_LOCATION 1

std::vector<DWORD>* fatal_nearest_warn(std::vector<DWORD>* warn_dates,std::vector<DWORD>* fatal_dates){
	std::vector<DWORD>* result=new std::vector<DWORD>;
	unsigned int i=0,j=0;
	while(j<fatal_dates->size()){
		if(i==0&&warn_dates[0][i]>fatal_dates[0][j]){
			j++;
		}else{
			if(i+1==warn_dates->size()){
				for(j=j;j<fatal_dates->size();j++){
					result->push_back(fatal_dates[0][j]-warn_dates[0][i]);
				}
			}else{
				if(warn_dates[0][i+1]>fatal_dates[0][j]){
					result->push_back(fatal_dates[0][j]-warn_dates[0][i]);
					j++;
				}else{
					i++;
				}
			}
		}
	}
	return result;
}

std::vector<Interval>* temporal_clustering(std::vector<DWORD>* dates,DWORD threshold){
	std::vector<Interval>* result=new std::vector<Interval>(1);
	if(dates->size()==1){
		result[0][0].start=0;
		result[0][0].end=1;
		return result;
	}
	std::vector<DWORD>* diff=new std::vector<DWORD>(dates->size()-1);
	Interval interval;
	interval.start=0;
	result[0][0]=interval;
	WORD i;
	for(i=0;i<(WORD)diff->size();i++){
		diff[0][i]=dates[0][i+1]-dates[0][i];
	}
	for(i=0;i<(WORD)diff->size();i++){
		if(diff[0][i]>threshold){
			interval.start=i+1;
			result[0][result->size()-1].end=i+1;
			result->push_back(interval);
		}
	}
	result[0][result->size()-1].end=dates->size();
	delete diff;
	return result;
}
/*
 * Get a level of location.
 * Separator '-'.
 * The first location_level number of element separated by '-' is filtered out.
 * location_level is assumed to be a postive integer.
*/
char* extract_location_level(char* location,WORD location_level){
	if(location_level<1){
		char* result=Calloc(1,char);
		result[0]='\0';
		return result;
	}
	WORD i=0,count=0;

	while(location[i]!='\0'){
		if(location[i]=='-'){
			count++;
			if(location_level==count){
				break;
			}
		}
		i++;
	}
	char *result=Calloc(i+1,char);
	memcpy(result,location,sizeof(char)*i);
	result[i]='\0';
	return result;
}

/*
 * Get distance between two RAS event location.
 * 4: Different Rack
 * 3: Same Rack, different midplane
 * 2: Same Midplane, different nodeboard
 * 1: Same nodeboard, different node
 * 0: Same node
*/

DWORD location_distance(char* loc1,char* loc2){
	DWORD result=4;
	while(1){
		if(*loc1!=*loc2){
			return result;
		}
		if(*loc1=='-'){
			result--;
		}else if(*loc1=='\0'){
			return 0;
		}
		loc2++;
		loc1++;
	}
	return result;
}

/*
 * Join COMPONENT,CATEGORY,SEVERITY with comma.
*/
char* str_join(char* component,char* category,char* severity){
	int length=strlen(component)+strlen(category)+strlen(severity)+3;
	char* result=Calloc(length,char);
	result[0]='\0';
	strcat(result,component);
	strcat(result,"-");
	strcat(result,category);
	strcat(result,"-");
	strcat(result,severity);
	return result;
}

std::vector<std::vector<DWORD>*>* unlist_temporal_feature_index(std::vector<std::vector<std::vector<DWORD>*>*>* st_clusters){
	std::vector<std::vector<DWORD>*> *result=new std::vector<std::vector<DWORD>*>;
	unsigned i,j;
	for(i=0;i<st_clusters->size();i++){
		for(j=0;j<st_clusters[0][i]->size();j++){
			result->push_back(st_clusters[0][i][0][j]);
		}
	}
	return result;
}


DWORD st_cluster_size(std::vector<std::vector<std::vector<std::vector<DWORD>*>*>*>* st_clusters){
	unsigned i,j;
	DWORD result=0;
	for(i=0;i<st_clusters->size();i++){
		for(j=0;j<st_clusters[0][i]->size();j++){
			result+=st_clusters[0][i][0][j]->size();
		}
	}
	return result;
}

DWORD st_cluster_size_by_date(std::vector<std::vector<std::vector<std::vector<DWORD>*>*>*>* st_clusters,std::vector<DWORD> *dates,DWORD date_limit){
	unsigned i,j,k;
	DWORD result=0;
	for(i=0;i<st_clusters->size();i++){
		for(j=0;j<st_clusters[0][i]->size();j++){
			for(k=0;k<st_clusters[0][i][0][j]->size();k++){
				if(dates[0][st_clusters[0][i][0][j][0][k][0][0]]>date_limit){
					result++;
				}
			}
		}
	}
	return result;
}


std::map<std::vector<DWORD>*,DWORD>* cluster_ranking(std::vector<std::vector<std::vector<std::vector<DWORD>*>*>*>* st_clusters){
	std::map<std::vector<DWORD>*,DWORD>* result=new std::map<std::vector<DWORD>*,DWORD>;
	unsigned i,j,k;
	DWORD counter=1;
	for(i=0;i<st_clusters->size();i++){
		for(j=0;j<st_clusters[0][i]->size();j++){
			for(k=0;k<st_clusters[0][i][0][j]->size();k++){
				result[0][st_clusters[0][i][0][j][0][k]]=counter;
				counter++;
			}
		}
	}
	return result;
}

void normalize_features(std::vector<Feature> *features){
	DTYPE mean_interval_max=-1;
	DTYPE mean_interval_min=-1;
	DTYPE last_fatal_max=-1;
	DTYPE last_fatal_min=-1;
	unsigned i;
	for(i=0;i<features->size();i++){
		if(last_fatal_max==-1||last_fatal_max<features[0][i].last_fatal){
			last_fatal_max=features[0][i].last_fatal;
		}
		if(last_fatal_min==-1||last_fatal_min>features[0][i].last_fatal){
			last_fatal_min=features[0][i].last_fatal;
		}
		if(mean_interval_max==-1||mean_interval_max<features[0][i].mean_interval){
			mean_interval_max=features[0][i].mean_interval;
		}
		if( mean_interval_min==-1||mean_interval_min>features[0][i].mean_interval){
			 mean_interval_min=features[0][i].mean_interval;
		}
	}
	for(i=0;i<features->size();i++){
		if(features[0][i].last_fatal==-1){
			features[0][i].last_fatal=1;
		}else{
			features[0][i].last_fatal=(features[0][i].last_fatal-last_fatal_min)/(last_fatal_max-last_fatal_min);
		}
		features[0][i].mean_interval=(features[0][i].mean_interval-mean_interval_min)/(mean_interval_max-mean_interval_min);
	}
}

/*
void construct_empty_features(std::vector<DWORD>* cluster,std::vector<std::vector<std::vector<std::vector<DWORD>*>*>*>* fatal_clusters,std::vector<std::vector<char*>*>* table,std::vector<std::string>* attributes,std::vector<DWORD> *warn_dates,std::vector<DWORD> *fatal_dates,DWORD warn_max_count,DWORD stripe,std::vector<Feature>* result){

}
*/

void warn_fatal_bind(std::vector<std::vector<char*>*>* table,std::vector<DWORD>* warn_cluster, std::vector<DWORD>* fatal_cluster_start, std::vector<DWORD>* fatal_cluster_end, std::vector<DWORD>* fatal_cluster_location_count, std::vector<std::set<std::string>*>* fatal_cluster_locations, std::vector<DWORD> *warn_dates,std::vector<DWORD> *fatal_dates,std::vector<std::string>* attributes, std::vector<Feature>* result,DWORD stripe,DWORD warn_max_count,DWORD only_fatal,DWORD* insufficient_information){
	if(stripe+warn_max_count>=(int)warn_cluster->size()){
		if(only_fatal==0){
			insufficient_information[0]+=fatal_cluster_start->size();
		}
		return;
	}
	unsigned i,j,fatal_index=0;
	char* feature_name;
	Feature element;
	std::vector<DTYPE>* feature;
	std::vector<DTYPE>* intervals=new std::vector<DTYPE>;
	std::vector<DTYPE>* spatial_intervals=new std::vector<DTYPE>;
	DWORD next_window_end=stripe+warn_max_count,last_fatal=-1,mean_interval=0,total_warn_count=0,qualified_warn_count;
	WORD failure=0;
	DWORD location_match;
	std::set<std::string>::iterator it;
	std::vector<std::string>::iterator it2;
	std::map<DWORD,std::set<std::string>*>::iterator it3;
	DTYPE temp,s_dis;
	if(fatal_index<fatal_cluster_start->size()){
		last_fatal=fatal_dates[0][fatal_cluster_start[0][fatal_index]];
	}
	while(fatal_cluster_start->size()>0&&fatal_dates[0][fatal_cluster_start[0][fatal_index]]<warn_dates[0][warn_cluster[0][next_window_end]]){
		if(fatal_index<fatal_cluster_start->size()){
			last_fatal=fatal_dates[0][fatal_cluster_start[0][fatal_index]];
		}
		fatal_index++;
		if(only_fatal==0){
			insufficient_information[0]++;
		}
		if(fatal_index==fatal_cluster_start->size()){
			return;
		}
	}
	std::map<std::string,DWORD> *feature_count=new std::map<std::string,DWORD>;

	std::vector<std::string> *remove_list=new std::vector<std::string>;
	std::map<std::string,DWORD>* warn_locations=new std::map<std::string,DWORD>;
	std::map<std::string,DWORD>::iterator it4;

	std::vector<std::set<std::string>*>* warn_location_counts=new std::vector<std::set<std::string>*>(4);
	for(i=0;i<warn_location_counts->size();i++){
		warn_location_counts[0][i]=new std::set<std::string>;
	}
	for(i=stripe;i<warn_cluster->size();i++){
		if((DWORD)(i-stripe)%warn_max_count==0&&i-stripe>0){
			if(i+warn_max_count<warn_cluster->size()){
				next_window_end=i+warn_max_count;
			}else{
				next_window_end=warn_cluster->size()-1;
			}
			//Conditions: there are remaining fatals, the fatal happens before the next window (so we can use previous window warn events to predict them)
			if(fatal_index<fatal_cluster_start->size()&&warn_dates[0][warn_cluster[0][next_window_end]]>=fatal_dates[0][fatal_cluster_start[0][fatal_index]]){
				element.label=1;
				failure=1;
				location_match=0;
				total_warn_count=0;
				//printf("%ld\n",warn_locations->size());
				remove_list->clear();
				//Try to match the fatal locations.
				qualified_warn_count=0;
				for(it=fatal_cluster_locations[0][fatal_index]->begin();it!=fatal_cluster_locations[0][fatal_index]->end();++it){
					if(warn_locations->find(*it)!=warn_locations->end()&&(warn_locations[0][*it]+.0)/total_warn_count>=0){
						location_match++;
						remove_list->push_back(*it);
					}
				}
				for(it4=warn_locations->begin();it4!=warn_locations->end();++it4){
					if((it4->second+.0)/total_warn_count>=0){
						qualified_warn_count++;
					}
				}
				//Erase those fatal locations that can be deduced from the warn location. The remainings are unpredictable locations.
				for(it2=remove_list->begin();it2!=remove_list->end();++it2){
					fatal_cluster_locations[0][fatal_index]->erase(*it2);
				}
				//Compute predictable percentage of fatal locations.
				element.location_pinpoint=(location_match+.0)/fatal_cluster_location_count[0][fatal_index];
				element.location_recovery=(location_match+.0)/qualified_warn_count;
			}else{
				element.label=-1;
			}
			element.lead_time=-1;
			//Write down start date of the feature, useful for dividing test set.
			element.start_date=warn_dates[0][warn_cluster[0][i-1]];
			//Compute mean interval of cluster events.
			element.mean_interval=mean_interval;
			mean_interval=0;
			//How long since the last fatal event occurred before prediction window.
			if(last_fatal==-1||last_fatal>warn_dates[0][warn_cluster[0][i]]){
				element.last_fatal=-1;
			}else{
				element.last_fatal=warn_dates[0][warn_cluster[0][i]]-last_fatal;
			}
			if(element.label==-1&&failure==1){
				//The last fatal event cluster may span across to this window. If it ends before the start of this window and this window does not overlap with any new fatal clusters, set failure=0 (so no need to check this next time even if element.label=-1)
				if(fatal_dates[0][fatal_cluster_end[0][fatal_index-1]]<warn_dates[0][warn_cluster[0][i]]){
					failure=0;
				}
			}else if(only_fatal==0||element.label==1){
				if(element.label==1){
					//Compute lead time
					element.lead_time=fatal_dates[0][fatal_cluster_start[0][fatal_index]]-warn_dates[0][warn_cluster[0][i-1]];
					element.fatal_start_date=fatal_dates[0][fatal_cluster_start[0][fatal_index]];
				}else{
					element.fatal_start_date=-1;
					element.location_pinpoint=-1;
					element.location_recovery=-1;
				}
				//construct feature for the warn cluster in the previous window
				feature=new std::vector<DTYPE>(attributes->size()+2*(warn_max_count-1)+warn_location_counts->size());
				element.features=feature;
				for(j=0;j<attributes->size();j++){
					std::string s=attributes[0][j];
					if(feature_count->find(s)==feature_count->end()){
						feature[0][j]=0;
					}else{
						//feature[0][j]=feature_count[0][s]/((DTYPE)warn_max_count);
						feature[0][j]=feature_count[0][s];
						//std::cout<<result[0][i]<<"\n";
					}
				}
				for(j=0;j<(unsigned)(warn_max_count-1);j++){
					feature[0][j+attributes->size()]=intervals[0][j];
				}
				for(j=0;j<(unsigned)(warn_max_count-1);j++){
					feature[0][j+attributes->size()+warn_max_count-1]=spatial_intervals[0][j];
				}
				for(j=0;j<warn_location_counts->size();j++){
					feature[0][j+attributes->size()+2*(warn_max_count-1)]=warn_location_counts[0][j]->size();
				}
				if(only_fatal==0){
					result->push_back(element);
				}
			}
			while(element.label==1){
				fatal_index++;
				//Swipe through all fatal clusters occured in the next window.
				if(fatal_index<fatal_cluster_start->size()&&warn_dates[0][warn_cluster[0][next_window_end]]>=fatal_dates[0][fatal_cluster_start[0][fatal_index]]){
					element.lead_time=fatal_dates[0][fatal_cluster_start[0][fatal_index]]-warn_dates[0][warn_cluster[0][i-1]];
					last_fatal=fatal_dates[0][fatal_cluster_start[0][fatal_index]];
					//Figure out location features (how many we recovered?)
					location_match=0;
					qualified_warn_count=0;
					remove_list->clear();
					for(it=fatal_cluster_locations[0][fatal_index]->begin();it!=fatal_cluster_locations[0][fatal_index]->end();++it){
						if(warn_locations->find(*it)!=warn_locations->end()&&(warn_locations[0][*it]+.0)/total_warn_count>=0){
							location_match++;
							remove_list->push_back(*it);
						}
					}
					for(it4=warn_locations->begin();it4!=warn_locations->end();++it4){
						if((it4->second+.0)/total_warn_count>=0){
							qualified_warn_count++;
						}
					}
					for(it2=remove_list->begin();it2!=remove_list->end();++it2){
						fatal_cluster_locations[0][fatal_index]->erase(*it2);
					}
					element.location_pinpoint=(location_match+.0)/fatal_cluster_location_count[0][fatal_index];
					element.location_recovery=(location_match+.0)/qualified_warn_count;
					element.fatal_start_date=fatal_dates[0][fatal_cluster_start[0][fatal_index]];
					//construct another feature
					if(only_fatal==0){
						result->push_back(element);
					}
				}else{
					break;
				}
			}
			intervals->clear();
			spatial_intervals->clear();
			feature_count->clear();
			warn_locations->clear();
			total_warn_count=0;
			for(j=0;j<warn_location_counts->size();j++){
				warn_location_counts[0][j]->clear();
			}
			
			if(i+warn_max_count>=warn_cluster->size()){
				break;
			}
		}
		if((DWORD)(i-stripe)%warn_max_count>0){
			if(i+1<warn_cluster->size()&&(DWORD)(i-stripe+1)%warn_max_count!=0&&warn_dates[0][warn_cluster[0][i]]==warn_dates[0][warn_cluster[0][i+1]]){
				intervals->push_back(0);
			}else{
				intervals->push_back(warn_dates[0][warn_cluster[0][i]]-warn_dates[0][warn_cluster[0][i-1]]);
			}
			temp=5;
			j=1;
			while(j<=(i-stripe)%warn_max_count&&warn_dates[0][warn_cluster[0][i]]==warn_dates[0][warn_cluster[0][i-j]]){
				s_dis=location_distance(table[0][COL_LOCATION][0][warn_cluster[0][i]],table[0][COL_LOCATION][0][warn_cluster[0][i-j]]);
				if(s_dis<temp){
					temp=s_dis;
				}
				j++;
			}
			j=-1;
			while(i-j<warn_cluster->size()&&(DWORD)(i-stripe-j)%warn_max_count!=0&&warn_dates[0][warn_cluster[0][i]]==warn_dates[0][warn_cluster[0][i-j]]){
				s_dis=location_distance(table[0][COL_LOCATION][0][warn_cluster[0][i]],table[0][COL_LOCATION][0][warn_cluster[0][i-j]]);
				if(s_dis<temp){
					temp=s_dis;
				}
				j--;
			}
			s_dis=location_distance(table[0][COL_LOCATION][0][warn_cluster[0][i]],table[0][COL_LOCATION][0][warn_cluster[0][i-1]]);
			if(s_dis<temp){
				temp=s_dis;
			}
			spatial_intervals->push_back((DTYPE)temp);
			mean_interval+=warn_dates[0][warn_cluster[0][i]]-warn_dates[0][warn_cluster[0][i-1]];
		}
		feature_name=str_join(table[0][COL_COMPONENT][0][warn_cluster[0][i]],table[0][COL_CATEGORY][0][warn_cluster[0][i]],table[0][COL_SEVERITY][0][warn_cluster[0][i]]);
		std::string s(feature_name);
		if(feature_count->find(s)==feature_count->end()){
			feature_count[0][s]=1;
		}else{
			feature_count[0][s]+=1;
		}
		Free(feature_name);
		feature_name=extract_location_level(table[0][COL_LOCATION][0][warn_cluster[0][i]],CLUSTER_LOCATION);
		std::string s2(feature_name);
		total_warn_count++;
		if(warn_locations->find(s2)==warn_locations->end()){
			warn_locations[0][s2]=1;
		}else{
			warn_locations[0][s2]+=1;
		}
		Free(feature_name);

		for(j=0;j<warn_location_counts->size();j++){
			feature_name=extract_location_level(table[0][COL_LOCATION][0][warn_cluster[0][i]],j+1);
			std::string s3(feature_name);
			warn_location_counts[0][j]->insert(s3);
			Free(feature_name);
		}
	}
	if(fatal_index!=fatal_cluster_start->size()){
		printf("error %d!=%ld\n",fatal_index,fatal_cluster_start->size());
	}
	for(i=0;i<warn_location_counts->size();i++){
		delete warn_location_counts[0][i];
	}
	delete warn_location_counts;
	delete intervals;
	delete spatial_intervals;
	delete remove_list;
	delete feature_count;
	delete warn_locations;
}

std::vector<Feature> *feature_matching(std::vector<Interval>* warn_t_clusters,std::vector<Interval>* fatal_t_clusters,std::vector<std::vector<std::vector<std::vector<DWORD>*>*>*>* warn_clusters,std::vector<std::vector<std::vector<std::vector<DWORD>*>*>*>* fatal_clusters,std::vector<DWORD> *warn_dates,std::vector<DWORD> *fatal_dates,std::vector<std::vector<char*>*>* warn_table,std::vector<std::vector<char*>*>* fatal_table,std::vector<std::string>* attributes,WORD location_level,DWORD warn_max_count){
	std::vector<Feature> *result=new std::vector<Feature>;
	std::vector<std::vector<DWORD>*>* warn_s_clusters;
	std::vector<std::vector<DWORD>*>* fatal_s_clusters;
	std::vector<DWORD> *fatal_cluster_start=new std::vector<DWORD>;
	std::vector<DWORD> *fatal_cluster_end=new std::vector<DWORD>;
	std::vector<DWORD> *fatal_cluster_location_count=new std::vector<DWORD>;
	std::vector<std::set<std::string>*> *fatal_cluster_locations=new std::vector<std::set<std::string>*>;
	std::vector<std::vector<std::vector<std::vector<DWORD>*>*>*>* sub_fatal_clusters=new std::vector<std::vector<std::vector<std::vector<DWORD>*>*>*>;
	unsigned i,j,k,w,t;
	DWORD warn_start, warn_start2,warn_end,fatal_start;
	DWORD fatal_start2;
	char* fatal_location,*warn_location,*tmp_str;
	char* feature_name;
	std::map<std::string,DWORD> *feature_count=new std::map<std::string,DWORD>;
	DWORD fatal_recovery_count=0;
	std::vector<DTYPE>* fatal_recovery_rate=new std::vector<DTYPE>;
	std::vector<DTYPE>* fatal_locations=new std::vector<DTYPE>;
	//Iterate through all warning
	DWORD count=0,insufficient_information=0,temp,temp2;

	for(i=0;i<warn_clusters->size();++i){
		warn_start=warn_t_clusters[0][i].start;
		warn_end=warn_t_clusters[0][i].end;
		//Break a temporal warn cluster into spatio-temporal warn clusters.
		warn_s_clusters=unlist_temporal_feature_index(warn_clusters[0][i]);
		//printf("i=%d,%ld\n",i,warn_s_clusters->size());
		sub_fatal_clusters->clear();
		//Filter out the temporal fatal clusters we are interested (overlapping with the current temporal warn cluster.)
		for(j=0;j<fatal_clusters->size();++j){
			fatal_start=fatal_t_clusters[0][j].start;
			if(warn_dates[0][warn_start]<=fatal_dates[0][fatal_start]&&warn_dates[0][warn_end]>=fatal_dates[0][fatal_start]){
				sub_fatal_clusters->push_back(fatal_clusters[0][j]);
			}
		}
		//For every spatio-temporal warn cluster 
		for(w=0;w<warn_s_clusters->size();++w){
			//Initialize local variables.
			for(j=0;j<fatal_cluster_locations->size();j++){
				delete fatal_cluster_locations[0][j];
			}
			fatal_cluster_locations->clear();
			fatal_cluster_start->clear();
			fatal_cluster_end->clear();
			fatal_cluster_location_count->clear();
			Feature feature;
			warn_start2=warn_s_clusters[0][w][0][0];
			feature.label=0;
			feature.lead_time=-1;
			feature.last_fatal=-1;
			//Figure out the location of the current warn cluster.
			warn_location=extract_location_level(warn_table[0][COL_LOCATION][0][warn_start2],location_level);
			/*
			for(t=0;t<warn_s_clusters[0][w]->size();t++){
				feature_name=extract_location_level(warn_table[0][COL_LOCATION][0][warn_s_clusters[0][w][0][t]],location_level);
				if(strcmp(feature_name,warn_location)!=0){
					printf("warn critical_error,%s,%s\n",warn_location,feature_name);
				}
				Free(feature_name);
			}
			*/
			// For every related temporal fatal cluster
			for(j=0;j<sub_fatal_clusters->size();++j){
				fatal_s_clusters=unlist_temporal_feature_index(sub_fatal_clusters[0][j]);
				//For every spatio-temporal fatal cluster
				for(k=0;k<fatal_s_clusters->size();k++){
					//Get start time and location of the spatio-temporal fatal cluster
					fatal_start2=fatal_s_clusters[0][k][0][0];
					fatal_location=extract_location_level(fatal_table[0][COL_LOCATION][0][fatal_start2],location_level);
					/*
					for(t=0;t<fatal_s_clusters[0][k]->size();t++){
						feature_name=extract_location_level(fatal_table[0][COL_LOCATION][0][fatal_s_clusters[0][k][0][t]],location_level);
						if(strcmp(feature_name,fatal_location)!=0){
							printf("fatal critical_error,%s,%s\n",fatal_location,feature_name);
						}
						Free(feature_name);
					}
					*/
					//If the fatal and warn clusters are at the same location and the fatal cluster overlaps and happen after the warn cluster. i.e this fatal can be predicted.
					if(strcmp(warn_location,fatal_location)==0&&fatal_dates[0][fatal_start2]<=warn_dates[0][warn_s_clusters[0][w][0][warn_s_clusters[0][w]->size()-1]]&&warn_dates[0][warn_start2]<=fatal_dates[0][fatal_start2]){
						//Count predictable fatal clusters.
						count++;
						//Store information for fatal cluster start/end (referenced later in feature construction)
						fatal_cluster_start->push_back(fatal_s_clusters[0][k][0][0]);
						fatal_cluster_end->push_back(fatal_s_clusters[0][k][0][fatal_s_clusters[0][k]->size()-1]);
						//Store predictable fatal cluster locations (referenced later in feature construction)
						std::set<std::string>* locations=new std::set<std::string>;
						for(t=0;t<fatal_s_clusters[0][k]->size();++t){
							tmp_str=extract_location_level(fatal_table[0][COL_LOCATION][0][fatal_s_clusters[0][k][0][t]],CLUSTER_LOCATION);
							std::string s(tmp_str);
							locations->insert(s);
							Free(tmp_str);
						}
						fatal_cluster_locations->push_back(locations);
						fatal_cluster_location_count->push_back(locations->size());
					}
					Free(fatal_location);
				}
				delete fatal_s_clusters;
			}
			if(fatal_cluster_start->size()>0||1>0){
				temp=0;
				//Construct features using fatal event locations, start/end time.
				warn_fatal_bind(warn_table,warn_s_clusters[0][w], fatal_cluster_start, fatal_cluster_end,fatal_cluster_location_count,fatal_cluster_locations,warn_dates,fatal_dates,attributes, result,0,warn_max_count,0,&temp);
				warn_fatal_bind(warn_table,warn_s_clusters[0][w], fatal_cluster_start, fatal_cluster_end,fatal_cluster_location_count,fatal_cluster_locations,warn_dates,fatal_dates,attributes, result,warn_max_count/2,warn_max_count,0,&temp2);
				insufficient_information+=temp;
				if(fatal_cluster_start->size()>0){
					for(j=temp;j<fatal_cluster_locations->size();++j){
						//fatal_recovery_rate->push_back(1.0-(.0+fatal_cluster_locations[0][j]->size())/fatal_cluster_location_count[0][j]);
						fatal_recovery_count+=fatal_cluster_location_count[0][j];
						fatal_recovery_rate->push_back(fatal_cluster_locations[0][j]->size());
						fatal_locations->push_back(fatal_cluster_location_count[0][j]);
					}
				}
			}else{
				//No fatal cluster overlaps with the current warn clsuter, construct current feature with 0.
				if((DWORD)warn_s_clusters[0][w]->size()>=warn_max_count){
					for(j=0;j<(unsigned)warn_max_count;++j){
						feature_name=str_join(warn_table[0][COL_COMPONENT][0][warn_s_clusters[0][w][0][j]],warn_table[0][COL_CATEGORY][0][warn_s_clusters[0][w][0][j]],warn_table[0][COL_SEVERITY][0][warn_s_clusters[0][w][0][j]]);
						std::string s(feature_name);
						if(feature_count->find(s)==feature_count->end()){
							feature_count[0][s]=1;
						}else{
							feature_count[0][s]+=1;
						}
						Free(feature_name);
					}
					feature.features=new std::vector<DTYPE>(attributes->size()+warn_max_count-1);
					for(j=0;j<attributes->size();++j){
						std::string s=attributes[0][j];
						if(feature_count->find(s)==feature_count->end()){
							feature.features[0][j]=0;
						}else{
							feature.features[0][j]=feature_count[0][s]/((DTYPE)warn_max_count);
							//std::cout<<result[0][i]<<"\n";
						}
					}
					for(j=0;j<(unsigned)(warn_max_count-1);j++){
						feature.features[0][j+attributes->size()]=warn_dates[0][warn_s_clusters[0][w][0][j+1]]-warn_dates[0][warn_s_clusters[0][w][0][j]];
					}
					result->push_back(feature);
				}
			}
			Free(warn_location);
		}
		delete warn_s_clusters;
	}
	//normalize_features(result);
	double fatal_size=(double)st_cluster_size(fatal_clusters);
	double fatal_recovery_mean=0,fatal_location_mean=0;
	for(i=0;i<fatal_recovery_rate->size();++i){
		fatal_recovery_mean+=fatal_recovery_rate[0][i];
		fatal_location_mean+=fatal_locations[0][i];
	}
	//fatal_recovery_mean/=fatal_recovery_rate->size();
	fatal_recovery_mean=1-fatal_recovery_mean/fatal_recovery_count;
	fatal_location_mean/=fatal_recovery_rate->size();
	printf("total_features=%ld,fatal_size=%d,unknown cause=%lf,insufficient information=%lf,fatal location recovery=%lf,fatal location size=%lf\n",result->size(),(DWORD)fatal_size,(fatal_size-count)/fatal_size,insufficient_information/fatal_size,fatal_recovery_mean,fatal_location_mean);
	delete sub_fatal_clusters;
	delete fatal_cluster_start;
	delete fatal_cluster_end;
	delete fatal_cluster_locations;
	delete fatal_cluster_location_count;
	delete fatal_recovery_rate;
	return result;
}

void delete_cluster_containers(std::vector<std::vector<std::vector<std::vector<DWORD>*>*>*>* st_clusters){
	unsigned i,j;
	for(i=0;i<st_clusters->size();i++){
		for(j=0;j<st_clusters[0][i]->size();j++){
			delete st_clusters[0][i][0][j];
		}
		delete st_clusters[0][i];		
	}
	delete st_clusters;
}

std::vector<std::vector<DWORD>*>* unlist_feature_index(std::vector<std::vector<std::vector<std::vector<DWORD>*>*>*>* st_clusters){
	std::vector<std::vector<DWORD>*> *result=new std::vector<std::vector<DWORD>*>;
	unsigned i,j,k;
	for(i=0;i<st_clusters->size();i++){
		for(j=0;j<st_clusters[0][i]->size();j++){
			for(k=0;k<st_clusters[0][i][0][j]->size();k++){
				result->push_back(st_clusters[0][i][0][j][0][k]);
			}
		}
	}
	return result;
}

void delete_st_clusters(std::vector<std::vector<std::vector<std::vector<DWORD>*>*>*>* st_clusters){
	unsigned i,j,k;
	for(i=0;i<st_clusters->size();i++){
		for(j=0;j<st_clusters[0][i]->size();j++){
			for(k=0;k<st_clusters[0][i][0][j]->size();k++){
				delete st_clusters[0][i][0][j][0][k];
			}
		}
	}
	delete st_clusters;
}

//The following code are for streaming features.

std::vector<DTYPE>* extract_cluster_features(std::vector<std::vector<char*>*>* table,std::vector<std::string>* attributes, std::vector<int>* critical_attribute_index, DWORD start, DWORD end){
	unsigned i,index;
	char* feature_name;
	std::map<std::string,DWORD> *feature_count=new std::map<std::string,DWORD>;
	std::vector<DTYPE>* result=new std::vector<DTYPE>(attributes->size());
	//Count number of features for every tuple (COMPONENT,CATEGORY,SEVERITY)
	for(i=start;i<(unsigned)end;i++){
		index=i;
		feature_name=str_join(table[0][COL_COMPONENT][0][index],table[0][COL_CATEGORY][0][index],table[0][COL_SEVERITY][0][index]);
		std::string s(feature_name);
		//std::cout<<s<<"\n";
		if(feature_count->find(s)==feature_count->end()){
			feature_count[0][s]=1;
		}else{
			feature_count[0][s]+=1;
		}
		Free(feature_name);
	}
	for(i=0;i<attributes->size();i++){
		std::string s=attributes[0][i];
		if (feature_count->find(s)==feature_count->end()){
			result[0][i]=0;
		}else{
			result[0][i]=feature_count[0][s]/(end-start+0.0);
			//std::cout<<result[0][i]<<"\n";
		}
	}
        int check = 0;
	for (i=0; i < critical_attribute_index->size(); i++){
		if (result[0][critical_attribute_index[0][i]]>0){
			check = 1;
			break;
		}
	}
	if (check==0){
		delete result;
		result = NULL;
	}
	delete feature_count;
	return result;
}

std::vector<FatalCluster>* st_cluster_to_features(std::vector<Interval>* st_clusters,std::vector<std::vector<char*>*>* table,std::vector<DWORD> *dates,std::vector<std::string>* attributes,std::vector<std::string> *filter_attributes){
	std::vector<FatalCluster>* result=new std::vector<FatalCluster>;
	FatalCluster f;
	unsigned i,j;
	int check;
        std::vector<int>* critical_attribute_index = new std::vector<int>;
	for (i=0; i<attributes->size(); i++){
		check = 0;
		for (j=0; j<filter_attributes->size(); j++){
			if (attributes[0][i].find(filter_attributes[0][j]) != std::string::npos){
				check =1;
				break;
        		}
		}
		if (check==1){
			critical_attribute_index->push_back(i);
		}
	}
	for(i=0;i<st_clusters->size();i++){
		f.features=extract_cluster_features(table,attributes,critical_attribute_index, st_clusters[0][i].start,st_clusters[0][i].end);
		if (f.features!=NULL){
                        f.start=st_clusters[0][i].start;
                        f.end=st_clusters[0][i].end;
			f.start_date=dates[0][st_clusters[0][i].start];
			f.end_date=dates[0][st_clusters[0][i].end-1];
			f.event_size=st_clusters[0][i].end-st_clusters[0][i].start;
			result->push_back(f);
		}
	}
	delete critical_attribute_index;
	return result;
}

/*
 * Construct a map that contains all spatial clusters.
*/
std::map<std::string,std::vector<WORD>*> *filtering_index_by_location(std::vector<char*> *locations,WORD start,WORD end,WORD location_level){
	std::map<std::string,std::vector<WORD>*> *result=new std::map<std::string,std::vector<WORD>*>;
	DWORD i;
	for(i=start;i<end;i++){
		char* extracted_string=extract_location_level(locations[0][i],location_level);
		std::string s=std::string(extracted_string);
		if(result->find(s)==result->end()){
			std::vector<WORD>* v=new std::vector<WORD>;
			v->push_back(i);
			result[0][s]=v;
		}else{
			result[0][s]->push_back(i);
		}
		Free(extracted_string);
	}
	return result;
}
/*
 A more general version of temporal clustering that apply to interleaved index within an array.
*/
std::vector<Interval>* temporal_clustering_with_index(std::vector<DWORD>* dates,std::vector<WORD>* index,DWORD threshold){
	std::vector<Interval>* result=new std::vector<Interval>(1);
	if(index->size()==1){
		result[0][0].start=0;
		result[0][0].end=1;
		return result;
	}
	std::vector<DWORD>* diff=new std::vector<DWORD>(index->size()-1);
	Interval interval;
	interval.start=0;
	result[0][0]=interval;
	WORD i;
	for(i=0;i<(WORD)diff->size();i++){
		diff[0][i]=dates[0][index[0][i+1]]-dates[0][index[0][i]];
	}
	for(i=0;i<(WORD)diff->size();i++){
		if(diff[0][i]>threshold){
			interval.start=i+1;
			result[0][result->size()-1].end=i+1;
			result->push_back(interval);
		}
	}
	result[0][result->size()-1].end=index->size();
	delete diff;
	return result;
}


std::vector<std::vector<std::vector<std::vector<DWORD>*>*>*>* cluster_filter(std::vector<Interval>* t_clusters,std::vector<DWORD> *dates,std::vector<char*>* locations,DWORD threshold,WORD location_level){
	WORD i,j,k,w,start,end,sub_start,sub_end;
	std::map<std::string,std::vector<WORD>*> *index_map;
	std::vector<std::vector<std::vector<std::vector<DWORD>*>*>*>* result=new std::vector<std::vector<std::vector<std::vector<DWORD>*>*>*>(t_clusters->size());
	std::vector<Interval>* sub_t_clusters;
	std::vector<WORD>* index;
	//For each of the temporal clusters
	for(i=0;i<(WORD)t_clusters->size();i++){
		start=t_clusters[0][i].start;
		end=t_clusters[0][i].end;
		//printf("start=%d,end=%d\n",start,end);
		//Extract unique locations
		index_map=filtering_index_by_location(locations,start,end,location_level);
		result[0][i]=new std::vector<std::vector<std::vector<DWORD>*>*>(index_map->size());
		std::map<std::string,std::vector<WORD>*>::iterator it;
		j=0;
		//For each of the locations
		for(it=index_map->begin();it!=index_map->end();it++){
			//Separate index for this particular location
			index=it->second;
			//Sub-temporal clustering at this location
			sub_t_clusters=temporal_clustering_with_index(dates,index,threshold);
			result[0][i][0][j]=new std::vector<std::vector<DWORD>*>(sub_t_clusters->size());
			for(k=0;k<(WORD)sub_t_clusters->size();k++){
				//map sub-temporal clusters index to global index.
				sub_start=sub_t_clusters[0][k].start;
				sub_end=sub_t_clusters[0][k].end;
				result[0][i][0][j][0][k]=new std::vector<DWORD>(sub_end-sub_start);
				for(w=sub_start;w<sub_end;w++){
					result[0][i][0][j][0][k][0][w-sub_start]=index[0][w];
				}
			}
			delete sub_t_clusters;
			delete index;
			j++;
		}
		delete index_map;
	}
	return result;
}

void construct_streaming_feature(std::vector<FatalCluster>* fatal_clusters,std::vector<DWORD>* warn_dates,std::vector<DWORD>* attribute_column,std::vector<std::vector<DWORD>*> *location_columns,std::vector<std::vector<DWORD>*> *fatal_location_columns,DWORD attribute_size,DWORD start,DWORD end,DWORD window_size,DWORD fatal_index,DWORD lead_time_size, StreamingFeature* feature){
	std::vector<DWORD> *feature_count=new std::vector<DWORD>(attribute_size);
	std::fill(feature_count->begin(),feature_count->end(),0);
	DWORD i,j,k;
	std::vector<std::set<DWORD>*> *location_sets=new std::vector<std::set<DWORD>*>(4);
	std::vector<std::set<DWORD>*> *fatal_location_sets=new std::vector<std::set<DWORD>*>(4);
	std::set<DWORD>::iterator it;
	for(i=0;i<4;i++){
		location_sets[0][i]=new std::set<DWORD>;
		fatal_location_sets[0][i]=new std::set<DWORD>;
	}
	for(i=start;i<end;i++){
		feature_count[0][attribute_column[0][i]]+=1;
		for(j=0;j<4;j++){
			location_sets[0][j]->insert(location_columns[0][j][0][i]);
		}
	}
	feature->start_date=warn_dates[0][start];
	feature->time_span=warn_dates[0][end-1]-warn_dates[0][start];
	feature->features=new std::vector<DTYPE>(attribute_size);
	feature->location_counts=new std::vector<DWORD>(4);
	feature->temporal_diff=new std::vector<DWORD>(window_size-1);
	for(i=0;i<window_size-1;i++){
		feature->temporal_diff[0][i]=warn_dates[0][i+1]-warn_dates[0][i];
	}
	for(i=0;i<attribute_size;++i){
		//feature->features[0][i]=feature_count[0][i]/(end-start+.0);
		feature->features[0][i]=feature_count[0][i];
	}
	for(i=0;i<4;i++){
		feature->location_counts[0][i]=location_sets[0][i]->size();
	}
	if(fatal_index>0){
		feature->last_fatal=warn_dates[0][end-1]-fatal_clusters[0][fatal_index-1].start_date;
	}else{
		feature->last_fatal=-1000;
	}
	feature->lead_times=new std::vector<DWORD>(lead_time_size);
	for(i=0;i<lead_time_size;++i){
		if(fatal_index<(DWORD)fatal_clusters->size()){
			feature->lead_times[0][i]=fatal_clusters[0][fatal_index].start_date-warn_dates[0][end-1];
			if (i==0){
				feature->fatal_index=fatal_index;
				for(j=fatal_clusters[0][fatal_index].start;j<fatal_clusters[0][fatal_index].end;j++){
					for(k=0;k<4;k++){
						fatal_location_sets[0][k]->insert(fatal_location_columns[0][k][0][j]);
					}
				}
				feature->location_recovery=new std::vector<DTYPE>(4);
				for(j=0;j<4;j++){
					feature->location_recovery[0][j]=fatal_location_sets[0][j]->size();
					for(it=location_sets[0][j]->begin();it!=location_sets[0][j]->end();++it){
						fatal_location_sets[0][j]->erase(*it);
					}
					feature->location_recovery[0][j]=1-(fatal_location_sets[0][j]->size()+.0)/feature->location_recovery[0][j];
				}
			}
			
		}else{
			feature->lead_times[0][i]=-1;
			feature->fatal_index=-1;
			feature->location_recovery=NULL;
		}
		fatal_index++;
	}
	for(i=0;i<4;i++){
		delete location_sets[0][i];
		delete fatal_location_sets[0][i];
	}
	delete location_sets;
	delete fatal_location_sets;
	delete feature_count;
}

/*
 * This function creates a map from an array of strings to an array of distinct integers
*/
std::map<std::string,DWORD>* convert_attribute_table(std::vector<std::string>* attributes){
	std::map<std::string,DWORD> *result=new std::map<std::string,DWORD>;
	unsigned i;
	for(i=0;i<attributes->size();i++){
		result[0][attributes[0][i]]=i;
	}
	return result;
}

/*
 * This function convert three attributes column in string to an array of distinct integers.
*/
std::vector<DWORD>* extract_attribute_column(std::vector<std::vector<char*>*>* warn_table,std::map<std::string,DWORD>* attribute_table){
	std::vector<DWORD> *result=new std::vector<DWORD>(warn_table[0][0]->size());
	char* feature_name;
	unsigned i;
	for(i=0;i<warn_table[0][0]->size();i++){
		feature_name=str_join(warn_table[0][COL_COMPONENT][0][i],warn_table[0][COL_CATEGORY][0][i],warn_table[0][COL_SEVERITY][0][i]);
		std::string s(feature_name);
		result[0][i]=attribute_table[0][s];
		Free(feature_name);
	}
	return result;
}

std::vector<DWORD>* extract_location_column(std::vector<std::vector<char*>*>* warn_table,std::map<std::string,DWORD>* attribute_table,DWORD location_level){
	std::vector<DWORD> *result=new std::vector<DWORD>(warn_table[0][0]->size());
	char* feature_name;
	unsigned i;
	for(i=0;i<warn_table[0][0]->size();i++){
		feature_name=extract_location_level(warn_table[0][COL_LOCATION][0][i],location_level);
		std::string s(feature_name);
		result[0][i]=attribute_table[0][s];
		Free(feature_name);
	}
	return result;
}

std::map<std::string,DWORD>* convert_location_table(std::vector<char*>* locations_col,DWORD location_level){
	std::set<std::string> *locations=new std::set<std::string>;
	std::map<std::string,DWORD> *result=new std::map<std::string,DWORD>;
	unsigned i;
	char* feature_name;
	for(i=0;i<locations_col->size();i++){
		feature_name=extract_location_level(locations_col[0][i],location_level);
		std::string s(feature_name);
		locations->insert(s);
		Free(feature_name);
	}
	printf("total number of locations=%ld\n",locations->size());
	std::set<std::string>::iterator it;
	i=0;
	for(it=locations->begin();it!=locations->end();++it){
		result[0][*it]=i;
		i++;
	}
	delete locations;
	return result;
}


std::vector<StreamingFeature>* streaming_feature(std::vector<std::vector<char*>*>* warn_table, std::vector<std::vector<char*>*>* fatal_table, std::vector<DWORD>* warn_dates,std::vector<std::string>* attributes,std::vector<FatalCluster>* fatal_clusters,DWORD lead_time_size,DWORD window_size){
	std::vector<StreamingFeature>* results=new std::vector<StreamingFeature>;
	DWORD i=0;
	DWORD start,end,size;
	StreamingFeature sf;
	DWORD fatal_index=0;
	DTYPE recovery_rate=0;
	DWORD recovery_count=0;
	//Optimization for joint distribution of attributes by converting string names to integers for fast access.
	std::map<std::string,DWORD> *attribute_table=convert_attribute_table(attributes);

	std::vector<DWORD> *attribute_col=extract_attribute_column(warn_table,attribute_table);
	delete attribute_table;
	//Optimization for joint distribution of locations by converting string names to integers for fast access.
	std::vector<std::vector<DWORD>*> *location_columns=new std::vector<std::vector<DWORD>*>(4);
	std::vector<std::vector<DWORD>*> *fatal_location_columns=new std::vector<std::vector<DWORD>*>(4);
	for(i=0;i<4;i++){
		printf("Processing location %d conversion\n",i+1);
		attribute_table=convert_location_table(warn_table[0][COL_LOCATION],i+1);
		location_columns[0][i]=extract_location_column(warn_table,attribute_table,i+1);
		delete attribute_table;
		attribute_table=convert_location_table(fatal_table[0][COL_LOCATION],i+1);
		fatal_location_columns[0][i]=extract_location_column(fatal_table,attribute_table,i+1);
		delete attribute_table;
	}

	std::set<DWORD> *covered_index=new std::set<DWORD>;
	i=0;
	size=0;
	start=0;
	while(i<(DWORD)warn_dates->size()){
		if(size==window_size){
			end=i;
			//For the last timestamp, fill all events for exactly the same time stamp.
			while(end<(DWORD)warn_dates->size()&&(warn_dates[0][end]-warn_dates[0][i])<60){
				end++;
			}
			//Flush fatal index (so only fatals that happen after the events used for features are considered.)
			while(fatal_index<(DWORD)fatal_clusters->size()&&fatal_clusters[0][fatal_index].start_date<warn_dates[0][i]){
				fatal_index++;
			}
			//Build features
			construct_streaming_feature(fatal_clusters,warn_dates,attribute_col,location_columns,fatal_location_columns,attributes->size(),start,end,window_size,fatal_index,lead_time_size,&sf);
			results->push_back(sf);
			if(sf.fatal_index!=-1){
				covered_index->insert(sf.fatal_index);
				if(sf.lead_times[0][0]<3600){
					recovery_rate+=sf.location_recovery[0][0];
					recovery_count++;
				}
			}
			//Flush one time stamp (we construct features starting from the next time stamp)
			i=start;
			while(i<(DWORD)warn_dates->size()-1&&warn_dates[0][i]-warn_dates[0][start]<60&&i<end){
				i++;
			}
			//printf("%d\n",i);
			start=i;
			size=0;
		}else{
			size++;
			i++;
		}
	}
	for(i=0;i<4;i++){
		delete location_columns[0][i];
	}
	printf("total features=%ld, covered fatal=%ld, total fatal=%ld,location recovery=%lf\n",results->size(), covered_index->size(),fatal_clusters->size(),recovery_rate/recovery_count);
	delete location_columns;
	delete fatal_location_columns;
	delete covered_index;
	delete attribute_col;
	return results;
}


