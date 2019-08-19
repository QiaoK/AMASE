#include "data_functions.h"

int main(int argc,char** argv){
	DWORD window_size,lead_time_count,t_cluster_threshold;
	std::vector<std::vector<char*>*>* copy_table;
	std::vector<std::string>* temp;
	if(argc!=6){
		printf("Usage:./streaming_feature filename1 filename2 fatal_t_threshold window_size lead_time_count\n");
		return 1;
	}
	std::vector<std::string>* attributes=read_attributes("sys_attributes.txt");
	std::vector<std::string>* fatal_attributes=read_attributes("fatal_attributes.txt");
	std::vector<std::string>* filter_attributes=read_attributes("filter_attributes.txt");
	temp = attributes;
	attributes = filter_attributes_by_keywords(attributes, filter_attributes);
	delete temp;
	temp = fatal_attributes;
	fatal_attributes = filter_attributes_by_keywords(fatal_attributes, filter_attributes);
	delete temp;
	printf("attribute read and filter completed\n");

	std::vector<std::vector<char*>*>* warn_table=read_table(argv[1]);
	printf("# of warn is %ld\n",warn_table[0][0]->size());
        copy_table = filter_events(warn_table, filter_attributes);
	delete_table(warn_table);
	warn_table = copy_table;
	std::vector<std::vector<char*>*>* fatal_table=read_table(argv[2]);
	printf("# of fatal is %ld\n",fatal_table[0][0]->size());
        copy_table = filter_events(fatal_table, filter_attributes);

	delete_table(fatal_table);
	fatal_table = copy_table;

	printf("Table filter finished, # of warn is %ld, # of fatal is %ld\n",warn_table[0][1]->size(),fatal_table[0][1]->size());

	std::vector<DWORD>* warn_dates=chars2ints(warn_table[0][COL_DATE]);
	std::vector<DWORD>* fatal_dates=chars2ints(fatal_table[0][COL_DATE]);

        printf("check %ld, %ld\n",warn_dates->size(),fatal_dates->size());
	t_cluster_threshold=atoi(argv[3]);	
	window_size=atoi(argv[4]);
	lead_time_count=atoi(argv[5]);
	printf("cluster h=%d, window size=%d,lead_times=%d\n",t_cluster_threshold,window_size,lead_time_count);

	printf("Data read finished\n");
	std::vector<Interval>* fatal_t_clusters=temporal_clustering(fatal_dates,t_cluster_threshold);
        printf("fatal clustering finished\n");
	std::vector<FatalCluster>* fatal_features=st_cluster_to_features(fatal_t_clusters,fatal_table,fatal_dates, fatal_attributes,filter_attributes);
	printf("total number of fatal_features=%ld, size of attributes=%ld, size of fatal attributes = %ld\n",fatal_features->size(),attributes->size(), fatal_attributes->size());
	printf("ST clustering for fatal events finished\n");

	std::vector<StreamingFeature>* features=streaming_feature(warn_table,fatal_table, warn_dates,attributes,fatal_features,lead_time_count,window_size);
	printf("Streaming feature construction finished\n");
	char filename[200];
	sprintf(filename,"streaming_features_%d_%d_%d.csv",window_size,t_cluster_threshold,lead_time_count);
	write_streaming_features(filename,features,attributes,lead_time_count);
	printf("Streaming feature writeback finished\n");

	delete attributes;
	delete fatal_attributes;
	delete filter_attributes;

	delete warn_dates;
	delete fatal_dates;
	delete_table(warn_table);
	delete_table(fatal_table);
	return 0;
}
