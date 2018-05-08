#include "data_functions.h"

int main(int argc,char** argv){
	WORD t_cluster_threshold1,s_cluster_threshold1,t_cluster_threshold2,s_cluster_threshold2,location_level,warn_max_count;
	if(argc!=9){
		printf("Usage:./st_clustering filename1 filename2 t_cluster_threshold1 t_cluster_threshold2 s_cluster_threshold1 s_cluster_threshold2 location_level warn_max_count\n");
		return 1;
	}
	t_cluster_threshold1=atoi(argv[3]);
	t_cluster_threshold2=atoi(argv[4]);
	s_cluster_threshold1=atoi(argv[5]);
	s_cluster_threshold2=atoi(argv[6]);
	location_level=atoi(argv[7]);
	warn_max_count=atoi(argv[8]);
	std::vector<std::vector<char*>*>* warn_table=read_table(argv[1]);
	std::vector<std::vector<char*>*>* fatal_table=read_table(argv[2]);
	std::vector<DWORD>* warn_dates=chars2ints(warn_table[0][COL_DATE]);
	std::vector<DWORD>* fatal_dates=chars2ints(fatal_table[0][COL_DATE]);
	printf("Data read finished\n");
	std::vector<Interval>* warn_t_clusters=temporal_clustering(warn_dates,t_cluster_threshold1);
	std::vector<Interval>* fatal_t_clusters=temporal_clustering(fatal_dates,t_cluster_threshold2);
	std::vector<std::string>* attributes=read_attributes("info_warn_attributes.txt");
	printf("Temporal clustering finished, warn time interval size=%ld, fatal time interval size=%ld\n",warn_t_clusters->size(),fatal_t_clusters->size());
	std::vector<std::vector<std::vector<std::vector<DWORD>*>*>*>* warn_st_clusters=cluster_filter(warn_t_clusters,warn_dates,warn_table[0][COL_LOCATION],s_cluster_threshold1,location_level);
	std::vector<std::vector<std::vector<std::vector<DWORD>*>*>*>* fatal_st_clusters=cluster_filter(fatal_t_clusters,fatal_dates,fatal_table[0][COL_LOCATION],s_cluster_threshold2,location_level);
	printf("Spatio-temporal clustering finished, warn st cluster size=%d, fatal st cluster size=%d\n",st_cluster_size(warn_st_clusters),st_cluster_size(fatal_st_clusters));
	std::vector<Feature> *result=feature_matching(warn_t_clusters,fatal_t_clusters,warn_st_clusters,fatal_st_clusters,warn_dates,fatal_dates,warn_table,fatal_table,attributes,location_level,warn_max_count);
	printf("Feature matching finished\n");
	char result_file_name[2000];
	sprintf(result_file_name,"st_features_%d_%d_%d_%d_%d.csv",t_cluster_threshold1,t_cluster_threshold2,s_cluster_threshold1,s_cluster_threshold2,location_level);
	write_features(result,result_file_name,attributes);
	printf("Feature write to local file finished\n");
	char filename[200];
	std::vector<std::string>* fatal_attributes=read_attributes("fatal_attributes.txt");
	std::vector<std::vector<DTYPE>*>* fatal_features=st_cluster_to_features(fatal_st_clusters,fatal_table,fatal_attributes);
	sprintf(filename,"fatal_features_%d_%d_%d.csv",t_cluster_threshold2,s_cluster_threshold2,location_level);
	write_clusters(filename,fatal_features,fatal_attributes);
	printf("fatal feature table write finished\n");
	delete warn_t_clusters;
	delete fatal_t_clusters;
	delete warn_dates;
	delete fatal_dates;
	delete attributes;
	delete fatal_attributes;
	delete_st_clusters(warn_st_clusters);
	delete_st_clusters(fatal_st_clusters);
	delete_table(warn_table);
	delete_table(fatal_table);
	return 0;
}
