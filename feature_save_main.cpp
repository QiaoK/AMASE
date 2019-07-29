#include "data_functions.h"

int main(int argc,char** argv){
	WORD t_cluster_threshold1,s_cluster_threshold1,location_level;
	if(argc!=5){
		printf("Usage:./fatal_features filename t_cluster_threshold s_cluster_threshold location_level\n");
		return 1;
	}
	t_cluster_threshold1=atoi(argv[2]);
	s_cluster_threshold1=atoi(argv[3]);
	location_level=atoi(argv[4]);

	std::vector<std::vector<char*>*>* fatal_table=read_table(argv[1]);
	std::vector<DWORD>* fatal_dates=chars2ints(fatal_table[0][COL_DATE]);
	std::vector<Interval>* fatal_t_clusters=temporal_clustering(fatal_dates,t_cluster_threshold1);
	std::vector<std::string>* attributes=read_attributes("fatal_attributes.txt");
	std::vector<std::string>* filter_attributes=read_attributes("fatal_attributes.txt");
	std::vector<FatalCluster>* fatal_features=st_cluster_to_features(fatal_t_clusters,fatal_table,fatal_dates,attributes,filter_attributes);
	char filename[200];
	sprintf(filename,"fatal_features_%d_%d_%d.csv",t_cluster_threshold1,s_cluster_threshold1,location_level);
	write_clusters(filename,fatal_features,attributes);
	printf("total number of fatal_features=%ld, size of attributes=%ld\n",fatal_features->size(),attributes->size());
	delete fatal_dates;
	delete fatal_t_clusters;
	delete_table(fatal_table);
	delete attributes;
	delete filter_attributes;
}
