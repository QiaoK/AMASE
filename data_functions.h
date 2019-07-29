#ifndef DATA_FUNCTIONS_H
#define DATA_FUNCTIONS_H

#include "amase.h"


extern std::vector<std::vector<char*>*>* read_table(const char* filename);
extern std::vector<DWORD>* chars2ints(std::vector<char*>* inputs);
extern std::vector<std::string>* read_attributes(const char* filename);
extern void delete_table(std::vector<std::vector<char*>*>* table);
extern void write_features(std::vector<Feature> *features,const char* filename,std::vector<std::string>* attributes,DWORD max_count);
extern void write_clusters(const char* filename,std::vector<FatalCluster>* clusters,std::vector<std::string>* attributes);
extern void write_window_features(std::vector<Feature> *features,const char* filename,std::vector<std::string>* attributes);
#endif
