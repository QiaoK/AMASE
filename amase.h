#ifndef NN_H
#define NN_H
#include <vector>
#include <map>
#include <iostream>
#include <set>
#include <string>
#include <limits>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <algorithm>
#include <time.h>
#include <sys/time.h>
#include "random.h"
#define DWORD int
#define INTEGER int
#define WORD int
#define DTYPE float
#define TRUE 1
#define FALSE 0
#define BOOLEAN char
#define Calloc(a,b) ((b*)malloc(sizeof(b)*a))
#define Free free
#define MAX_DWORD std::numeric_limits<DWORD>::infinity()
#define MAX_DTYPE std::numeric_limits<DTYPE>::infinity()

#define COL_LOCATION 6
#define COL_DATE 3
#define COL_NUM 7
#define COL_CATEGORY 0
#define COL_COMPONENT 1
#define COL_SEVERITY 2

typedef struct{
	DWORD start;
	DWORD end;
}Interval;

typedef struct{
	std::vector<DTYPE> *features;
	DWORD label;
	DWORD lead_time;
}Feature;

extern std::vector<Interval>* temporal_clustering(std::vector<DWORD>* dates,DWORD threshold);
extern std::vector<std::vector<std::vector<std::vector<DWORD>*>*>*>* cluster_filter(std::vector<Interval>* t_clusters,std::vector<DWORD> *dates,std::vector<char*>* locations,DWORD threshold,WORD location_level);
extern std::vector<std::vector<DWORD>*>* unlist_feature_index(std::vector<std::vector<std::vector<std::vector<DWORD>*>*>*>* st_clusters);
extern void delete_cluster_containers(std::vector<std::vector<std::vector<std::vector<DWORD>*>*>*>* st_clusters);
extern std::vector<Feature> *feature_matching(std::vector<Interval>* warn_t_clusters,std::vector<Interval>* fatal_t_clusters,std::vector<std::vector<std::vector<std::vector<DWORD>*>*>*>* warn_clusters,std::vector<std::vector<std::vector<std::vector<DWORD>*>*>*>* fatal_clusters,std::vector<DWORD> *warn_dates,std::vector<DWORD> *fatal_dates,std::vector<std::vector<char*>*>* warn_table,std::vector<std::vector<char*>*>* fatal_table,std::vector<std::string>* attributes,WORD location_level,DWORD warn_max_count);
extern void delete_st_clusters(std::vector<std::vector<std::vector<std::vector<DWORD>*>*>*>* st_clusters);
extern DWORD st_cluster_size(std::vector<std::vector<std::vector<std::vector<DWORD>*>*>*>* st_clusters);
extern std::vector<std::vector<DTYPE>*>* st_cluster_to_features(std::vector<std::vector<std::vector<std::vector<DWORD>*>*>*>* st_clusters,std::vector<std::vector<char*>*>* table,std::vector<std::string>* attributes);
extern std::map<std::vector<DWORD>*,DWORD>* cluster_ranking(std::vector<std::vector<std::vector<std::vector<DWORD>*>*>*>* st_clusters);
#endif
