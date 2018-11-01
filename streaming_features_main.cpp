#include "data_functions.h"

int main(int argc,char** argv){
	DWORD window_size,lead_time_count,t_cluster_threshold;
	if(argc!=7){
		printf("Usage:./st_clustering filename1 filename2 fatal_t_threshold window_size lead_time_count location_level\n");
		return 1;
	}
	DWORD test_start_date=1462075200;
	std::vector<std::vector<char*>*>* warn_table=read_table(argv[1]);
	std::vector<std::vector<char*>*>* fatal_table=read_table(argv[2]);
	std::vector<DWORD>* warn_dates=chars2ints(warn_table[0][COL_DATE]);
	std::vector<DWORD>* fatal_dates=chars2ints(fatal_table[0][COL_DATE]);
	t_cluster_threshold=atoi(argv[3]);	
	window_size=atoi(argv[4]);
	lead_time_count=atoi(argv[5]);
	location_level=atoi(argv[6]);
	std::vector<std::string>* attributes=read_attributes("sys_attributes.txt");

	printf("Data read finished\n");
	std::vector<Interval>* fatal_t_clusters=temporal_clustering(fatal_dates,t_cluster_threshold);
	std::vector<Feature>* fatal_features=st_cluster_to_features(fatal_t_clusters,fatal_table,fatal_dates,attributes);
	printf("ST clustering for fatal events finished\n");

	std::vector<StreamingFeature>* features=streaming_feature(warn_table, warn_dates,attributes,fatal_clusters,lead_time_size,window_size);
	printf("Streaming feature construction finished\n");
	return 0;
}
