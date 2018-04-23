#include "amase.h"

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


std::vector<DTYPE>* extract_feature(std::vector<DWORD>* cluster,std::vector<std::vector<char*>*>* table,std::vector<std::string>* attributes,DWORD warn_start,DWORD fatal_start,std::vector<DWORD> *warn_dates,std::vector<DWORD> *fatal_dates,DWORD warn_max_count,DWORD* lead_time,DWORD* warn_event_count){
	unsigned i,index;
	char* feature_name;
	std::map<std::string,DWORD> *feature_count=new std::map<std::string,DWORD>;
	std::vector<DTYPE>* result=new std::vector<DTYPE>(attributes->size());
	//Count number of features for every tuple (COMPONENT,CATEGORY,SEVERITY)
	lead_time[0]=0;
	warn_event_count[0]=cluster->size();
	for(i=0;i<cluster->size();i++){
		index=cluster[0][i];
		if((DWORD)i==warn_max_count||(fatal_start>=0&&warn_dates[0][index]>fatal_dates[0][fatal_start])||warn_start>0){
			if((DWORD)i==warn_max_count){
				lead_time[0]=fatal_dates[0][fatal_start]-warn_dates[0][cluster[0][i-1]];
			}
			warn_event_count[0]=i;
			break;
		}
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
		if(feature_count->find(s)==feature_count->end()){
			result[0][i]=0;
		}else{
			result[0][i]=feature_count[0][s]/((DTYPE)cluster->size());
			//std::cout<<result[0][i]<<"\n";
		}
	}
	delete feature_count;
	return result;
}

/*
 * For the warn event cluster, use warn_max_count as sliding window to extract warn features. The label is either -1, which indicates the fatal should happen later, or 1, which indicates that the next window would have a fatal event occur.
 * intermediate!=1 means all subsections, intermediate=1 means only the subsection that triggers fatal is added.
*/

void extract_features(std::vector<DWORD>* cluster,std::vector<std::vector<char*>*>* table,std::vector<std::string>* attributes,DWORD intermediate,DWORD fatal_start,std::vector<DWORD> *warn_dates,std::vector<DWORD> *fatal_dates,DWORD warn_max_count,DWORD* lead_time,DWORD* warn_event_count,DWORD stripe,std::vector<Feature>* result){
	unsigned i,j,index;
	char* feature_name;
	std::map<std::string,DWORD> *feature_count=new std::map<std::string,DWORD>;
	std::vector<DTYPE>* feature=NULL;
	//Count number of features for every tuple (COMPONENT,CATEGORY,SEVERITY)
	lead_time[0]=0;
	warn_event_count[0]=cluster->size();
	Feature element;
	DWORD total=0;
	//Iterate through the current warn cluster (element means which in the dataframe.)
	for(i=stripe;i<cluster->size();i++){
		index=cluster[0][i];
		if((fatal_start>=0&&warn_dates[0][index]>fatal_dates[0][fatal_start])){
			warn_event_count[0]=i;
			break;
		}
		if((DWORD)i%warn_max_count==0&&i>0){
			//If for warn_max_count number of warn events, there is no fatal, we mark it with label -1.
			if(i/warn_max_count>1){
				if(intermediate!=1){
					element.features=feature;
					element.label=-1;
					element.lead_time=fatal_dates[0][fatal_start]-warn_dates[0][cluster[0][i-1]];
					result->push_back(element);
				}else{
					delete feature;
				}
			}
			lead_time[0]=warn_dates[0][cluster[0][i]]-warn_dates[0][cluster[0][i-warn_max_count]];
			feature=new std::vector<DTYPE>(attributes->size());
			for(j=0;j<attributes->size();j++){
				std::string s=attributes[0][j];
				if(feature_count->find(s)==feature_count->end()){
					feature[0][j]=0;
				}else{
					feature[0][j]=feature_count[0][s]/((DTYPE)total);
					//std::cout<<result[0][i]<<"\n";
				}
			}
			total=0;
			feature_count->clear();
			
		}
		feature_name=str_join(table[0][COL_COMPONENT][0][index],table[0][COL_CATEGORY][0][index],table[0][COL_SEVERITY][0][index]);
		std::string s(feature_name);
		//std::cout<<s<<"\n";
		if(feature_count->find(s)==feature_count->end()){
			feature_count[0][s]=1;
		}else{
			feature_count[0][s]+=1;
		}
		total++;
		Free(feature_name);
	}
	//printf("check\n");
	element.label=1;
	if(feature!=NULL){
		element.features=feature;
		result->push_back(element);
	}else if((int)i==warn_max_count){
		lead_time[0]=warn_dates[0][cluster[0][i]]-warn_dates[0][cluster[0][i-warn_max_count]];
		feature=new std::vector<DTYPE>(attributes->size());
		for(j=0;j<attributes->size();j++){
			std::string s=attributes[0][j];
			if(feature_count->find(s)==feature_count->end()){
				feature[0][j]=0;
			}else{
				feature[0][j]=feature_count[0][s]/((DTYPE)total);
				//std::cout<<result[0][i]<<"\n";
			}
		}
		element.features=feature;
		result->push_back(element);
	}
	/*else{
		feature=new std::vector<DTYPE>(attributes->size());
		for(j=0;j<attributes->size();j++){
			std::string s=attributes[0][j];
			if(feature_count->find(s)==feature_count->end()){
				feature[0][j]=0;
			}else{
				feature[0][j]=feature_count[0][s]/((DTYPE)total);
				//std::cout<<result[0][i]<<"\n";
			}
		}
		element.features=feature;
	}*/
	delete feature_count;
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

std::vector<Feature> *feature_matching(std::vector<Interval>* warn_t_clusters,std::vector<Interval>* fatal_t_clusters,std::vector<std::vector<std::vector<std::vector<DWORD>*>*>*>* warn_clusters,std::vector<std::vector<std::vector<std::vector<DWORD>*>*>*>* fatal_clusters,std::vector<DWORD> *warn_dates,std::vector<DWORD> *fatal_dates,std::vector<std::vector<char*>*>* warn_table,std::vector<std::vector<char*>*>* fatal_table,std::vector<std::string>* attributes,WORD location_level,DWORD warn_max_count){
	std::vector<Feature> *result=new std::vector<Feature>;
	std::vector<std::vector<DWORD>*>* warn_s_clusters;
	std::vector<std::vector<DWORD>*>* fatal_s_clusters;
	std::vector<std::vector<std::vector<std::vector<DWORD>*>*>*>* sub_fatal_clusters=new std::vector<std::vector<std::vector<std::vector<DWORD>*>*>*>;
	unsigned i,j,k,w;
	DWORD warn_start,warn_end,fatal_start;
	DWORD warn_start2,warn_end2,fatal_start2;
	char* fatal_location,*warn_location;
	//Iterate through all warning
	DWORD counter=0;
	std::vector<DWORD> *warn_counts=new std::vector<DWORD>;
	std::vector<DWORD> *lead_times=new std::vector<DWORD>;
	DWORD warn_event_count=0,lead_time=0,achieved_count=0;
	std::map<std::vector<DWORD>*,DWORD>* fatal_ranking=cluster_ranking(fatal_clusters);
	for(i=0;i<warn_clusters->size();i++){
		warn_start=warn_t_clusters[0][i].start;
		warn_end=warn_t_clusters[0][i].end;
		warn_s_clusters=unlist_temporal_feature_index(warn_clusters[0][i]);
		//printf("i=%d,%ld\n",i,warn_s_clusters->size());
		sub_fatal_clusters->clear();
		for(j=0;j<fatal_clusters->size();j++){
			fatal_start=fatal_t_clusters[0][j].start;
			if(warn_dates[0][warn_start]<=fatal_dates[0][fatal_start]&&warn_dates[0][warn_end]>=fatal_dates[0][fatal_start]){
				sub_fatal_clusters->push_back(fatal_clusters[0][j]);
			}
		}
		//printf("sub_fatal_clusters=%ld\n",sub_fatal_clusters->size());
		for(w=0;w<warn_s_clusters->size();w++){
			Feature feature;
			warn_start2=warn_s_clusters[0][w][0][0];
			warn_end2=warn_s_clusters[0][w][0][warn_s_clusters[0][w]->size()-1];
			feature.label=0;
			feature.lead_time=-1;
			warn_location=extract_location_level(warn_table[0][COL_LOCATION][0][warn_start2],location_level);
			for(j=0;j<sub_fatal_clusters->size();j++){			
				fatal_s_clusters=unlist_temporal_feature_index(sub_fatal_clusters[0][j]);
				for(k=0;k<fatal_s_clusters->size();k++){
					fatal_start2=fatal_s_clusters[0][k][0][0];
					if(warn_dates[0][warn_start2]<=fatal_dates[0][fatal_start2]&&warn_dates[0][warn_end2]>=fatal_dates[0][fatal_start2]){
						fatal_location=extract_location_level(fatal_table[0][COL_LOCATION][0][fatal_start2],location_level);
						if(strcmp(warn_location,fatal_location)==0){
							//printf("%d\n",i);
							extract_features(warn_s_clusters[0][w],warn_table,attributes,feature.label,fatal_start2,warn_dates,fatal_dates,warn_max_count,&lead_time,&warn_event_count,0,result);
							feature.label=1;
							if(warn_event_count>=warn_max_count){
								achieved_count++;
							}
							warn_counts->push_back(warn_event_count);
							lead_times->push_back(lead_time);
							
							extract_features(warn_s_clusters[0][w],warn_table,attributes,feature.label,fatal_start2,warn_dates,fatal_dates,warn_max_count,&lead_time,&warn_event_count,warn_max_count/2,result);
							warn_counts->push_back(warn_event_count);
							lead_times->push_back(lead_time);
							
							counter++;
							//k=fatal_s_clusters->size();
							//j=sub_fatal_clusters->size();
						}
						Free(fatal_location);
					}
				}
				delete fatal_s_clusters;
			}
			if(feature.label==0){
				feature.features=extract_feature(warn_s_clusters[0][w],warn_table,attributes,-1,-1,warn_dates,fatal_dates,warn_max_count,&lead_time,&warn_event_count);
				result->push_back(feature);
			}
			Free(warn_location);
		}
		delete warn_s_clusters;
	}
	std::sort(warn_counts->begin(),warn_counts->end());
	std::sort(lead_times->begin(),lead_times->end());
	w=0;
	for(i=0;i<lead_times->size();i++){
		if(lead_times[0][i]<=7200){
			w++;
		}
	}
	printf("lead time<3600, %d out of %ld\n",w,lead_times->size());
	printf("%d samples paired out of %ld warn events, %d samples reached %d warn count\n",counter,result->size(),achieved_count,warn_max_count);
	printf("warn counts per sample distribution, 10%%=%d, 25%%=%d, 50%%=%d, 75%%=%d, 90%%=%d\n", warn_counts[0][warn_counts->size()*.1], warn_counts[0][warn_counts->size()*.25], warn_counts[0][warn_counts->size()*.5], warn_counts[0][warn_counts->size()*.75], warn_counts[0][warn_counts->size()*.9]);
	printf("lead time per sample distribution, 10%%=%d, 25%%=%d, 50%%=%d, 75%%=%d, 90%%=%d\n", lead_times[0][lead_times->size()*.1], lead_times[0][lead_times->size()*.25], lead_times[0][lead_times->size()*.5], lead_times[0][lead_times->size()*.75], lead_times[0][lead_times->size()*.9]);
	delete lead_times;
	delete warn_counts;
	delete fatal_ranking;
	delete sub_fatal_clusters;
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

std::vector<DTYPE>* extract_cluster_features(std::vector<DWORD>* cluster,std::vector<std::vector<char*>*>* table,std::vector<std::string>* attributes){
	unsigned i,index;
	char* feature_name;
	std::map<std::string,DWORD> *feature_count=new std::map<std::string,DWORD>;
	std::vector<DTYPE>* result=new std::vector<DTYPE>(attributes->size());
	//Count number of features for every tuple (COMPONENT,CATEGORY,SEVERITY)
	for(i=0;i<cluster->size();i++){
		index=cluster[0][i];
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
		if(feature_count->find(s)==feature_count->end()){
			result[0][i]=0;
		}else{
			result[0][i]=feature_count[0][s]/((DTYPE)cluster->size());
			//std::cout<<result[0][i]<<"\n";
		}
	}
	delete feature_count;
	return result;
}

std::vector<std::vector<DTYPE>*>* st_cluster_to_features(std::vector<std::vector<std::vector<std::vector<DWORD>*>*>*>* st_clusters,std::vector<std::vector<char*>*>* table,std::vector<std::string>* attributes){
	std::vector<std::vector<DTYPE>*>* result=new std::vector<std::vector<DTYPE>*>;
	unsigned i,j,k;
	for(i=0;i<st_clusters->size();i++){
		for(j=0;j<st_clusters[0][i]->size();j++){
			for(k=0;k<st_clusters[0][i][0][j]->size();k++){
				result->push_back(extract_cluster_features(st_clusters[0][i][0][j][0][k],table,attributes));
			}
		}
	}
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

/*
std::vector<std::string>* read_attributes(const char* filename){
	std::vector<std::string>* result=new std::vector<std::string>;
	FILE* stream1=fopen(filename,"r");
	char c;
	std::string *s=new std::string;
	while((c=fgetc(stream1))!=EOF){
		if(c=='\n'){
			if(s->length()>0){
				result->push_back(*s);
			}
			s->clear();
		}else{
			s->push_back(c);
		}
	}
	delete s;
	fclose(stream1);
	return result;
}

int main(){
	unsigned i,j,k,w;
	std::vector<char*>* locations=new std::vector<char*>;
	char c1[200],c2[200],c3[200],c4[200],c5[200],c6[200],c7[200],c8[200],c0[200];
	strcpy(c0,"R21-M0-N09-O13");
	strcpy(c1,"R21-M1-N00-O15");
	strcpy(c2,"R21-M1-N06-O21");
	strcpy(c3,"R21-M0-N01-O01");
	strcpy(c4,"R21-M0-N12-O35");
	strcpy(c5,"R21-M0-N13-O15");
	strcpy(c6,"R21-M0-N04-O23");
	strcpy(c7,"R21-M0-N08-O35");
	strcpy(c8,"R21-M0-N09-O35");

	locations->push_back(c0);
	locations->push_back(c1);
	locations->push_back(c2);
	locations->push_back(c3);
	locations->push_back(c4);
	locations->push_back(c5);
	locations->push_back(c6);
	locations->push_back(c7);
	locations->push_back(c8);

	std::vector<DWORD>* dates=new std::vector<DWORD>;
	dates->push_back(10);
	dates->push_back(11);
	dates->push_back(12);
	dates->push_back(13);
	dates->push_back(20);
	dates->push_back(21);
	dates->push_back(22);
	dates->push_back(30);
	dates->push_back(31);
	std::vector<Interval>* t_clusters=temporal_clustering(dates,1);
	std::vector<std::vector<std::vector<std::vector<DWORD>*>*>*>* test=cluster_filter(t_clusters,dates,locations,1,2);

	std::vector<std::vector<DWORD>*>* st_cluster=unlist_feature_index(test);
	std::vector<std::string>* attributes=read_attributes("attributes.txt");

	for(i=0;i<test->size();i++){
		printf("i=%d\n",i);
		for(j=0;j<test[0][i]->size();j++){
			printf("  j=%d\n",j);
			for(k=0;k<test[0][i][0][j]->size();k++){
				printf("    k=%d\n      w=",k);
				for(w=0;w<test[0][i][0][j][0][k]->size();w++){
					printf("%d,",test[0][i][0][j][0][k][0][w]);
				}
				printf("\n");
			}
		}
	}

}*/
