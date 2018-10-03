#include "data_functions.h"
#include "random.h"
#include <math.h>

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
	if(s->length()>0){
		result->push_back(*s);
	}
	delete s;
	fclose(stream1);
	return result;
}

void read_table_row(char* line,std::vector<std::vector<char*>*>* inputs){
	char* cpt=line;
	DWORD i,size;
	//parse inputs
	for(i=0;i<COL_NUM-1;i++){
		line=cpt;
		size=0;
		while(*cpt!=','){
			cpt++;
			size++;
		}
		cpt++;
		char *x=Calloc(size+1,char);
		x[size]='\0';
		memcpy(x,line,sizeof(char)*size);
		inputs[0][i]->push_back(x);
	}
	//parse for last value
	size=0;
	line=cpt;
	while(*cpt!='\0'){
		cpt++;
		size++;
	}
	cpt++;
	char *x=Calloc(size+1,char);
	x[size]='\0';
	memcpy(x,line,sizeof(char)*size);
	inputs[0][COL_NUM-1]->push_back(x);
}

void delete_table(std::vector<std::vector<char*>*>* table){
	unsigned i;
	for(i=0;i<table->size();i++){
		delete(table[0][i]);
	}
	delete table;
}

std::vector<std::vector<char*>*>* read_table(const char* filename){
	FILE* stream1=fopen(filename,"r");
	FILE* stream2=fopen(filename,"r");
	DWORD length=0;
	char* line;
	char c;
	WORD i;
	std::vector<std::vector<char*>*>* inputs=new std::vector<std::vector<char*>*>(COL_NUM);
	for(i=0;i<COL_NUM;i++){
		inputs[0][i]=new std::vector<char*>;
	}
	while((c=fgetc(stream1))!=EOF){
		length++;
		if(c=='\n'&&length>=1){
			line=Calloc(length+1,char);
			fgets(line,length,stream2);
			//printf("%s\n",line);
			fgetc(stream2);
			line[length]='\0';
			read_table_row(line,inputs);
			length=0;
			Free(line);
		}
	}
	fclose(stream1);
	fclose(stream2);
	return inputs;
}

void write_clusters(const char* filename,std::vector<std::vector<DTYPE>*>* clusters,std::vector<std::string>* attributes){
	FILE* stream=fopen(filename,"w");
	unsigned i,j;
	for(i=0;i<attributes->size();i++){
		if(i<attributes->size()-1){
			fprintf(stream,"%s,",attributes[0][i].c_str());
		}else{
			fprintf(stream,"%s",attributes[0][i].c_str());
		}
	}
	fprintf(stream,"\n");
	for(i=0;i<clusters->size();i++){
		for(j=0;j<clusters[0][i]->size();j++){
			if(j<clusters[0][i]->size()-1){
				fprintf(stream,"%f,",clusters[0][i][0][j]);
			}else{
				fprintf(stream,"%f",clusters[0][i][0][j]);
			}
		}
		fprintf(stream,"\n");
	}
	fclose(stream);
}

void write_window_features(std::vector<Feature> *features,const char* filename,std::vector<std::string>* attributes){
	FILE* stream=fopen(filename,"w");
	unsigned i,j;
	for(i=0;i<attributes->size();i++){
		fprintf(stream,"%s,",attributes[0][i].c_str());
	}
	fprintf(stream,"LEAD_TIME,");
	fprintf(stream,"DATE,");
	fprintf(stream,"LOCATION_PINPOINT,");
	fprintf(stream,"LOCATION_RECOVERY,");
	fprintf(stream,"WARN_LOCATIONS,");
	fprintf(stream,"FATAL_LOCATIONS,");
	fprintf(stream,"FATAL_START_DATE,");
	fprintf(stream,"FATAL\n");
	for(i=0;i<features->size();i++){
		//printf("write line size=%ld\n",features[0][i].features->size());
		for(j=0;j<features[0][i].features->size();j++){
			if(features[0][i].features[0][j]==0){
				fprintf(stream,"%d,",0);
			}else{
				if(j<attributes->size()){
					fprintf(stream,"%4f,",features[0][i].features[0][j]);
				}else{
					fprintf(stream,"%d,",(DWORD)features[0][i].features[0][j]);
				}
			}
		}
		fprintf(stream,"%d,",features[0][i].lead_time);
		fprintf(stream,"%d,",features[0][i].start_date);
		fprintf(stream,"%4f,",features[0][i].location_pinpoint);
		fprintf(stream,"%4f,",features[0][i].location_recovery);
		fprintf(stream,"%d,",features[0][i].warn_location_size);
		fprintf(stream,"%d,",features[0][i].fatal_location_size);
		fprintf(stream,"%d,",features[0][i].fatal_start_date);
		fprintf(stream,"%d\n",features[0][i].label);
	}
}

void write_features(std::vector<Feature> *features,const char* filename,std::vector<std::string>* attributes,DWORD max_count){
	FILE* stream=fopen(filename,"w");
	unsigned i,j;
	for(i=0;i<attributes->size();i++){
		fprintf(stream,"%s,",attributes[0][i].c_str());
	}
	//printf("%ld\n",features[0][0].features->size());
	for(i=0;i<(unsigned)(max_count-1);i++){
		fprintf(stream,"TEMPORAL_INTERVAL_%05d,",i);
	}
	for(i=0;i<(unsigned)(max_count-1);i++){
		fprintf(stream,"SPATIAL_INTERVAL_%05d,",i);
	}
	for(i=0;i<4;i++){
		fprintf(stream,"SPATIAL_LOCATIONS_%05d,",i);
	}
	fprintf(stream,"MEAN_INTERVAL,");
	fprintf(stream,"LAST_FATAL,");
	fprintf(stream,"LEAD_TIME,");
	fprintf(stream,"DATE,");
	fprintf(stream,"LOCATION_PINPOINT,");
	fprintf(stream,"LOCATION_RECOVERY,");
	fprintf(stream,"FATAL_START_DATE,");
	fprintf(stream,"FATAL\n");
	for(i=0;i<features->size();i++){
		//printf("write line size=%ld\n",features[0][i].features->size());
		for(j=0;j<features[0][i].features->size();j++){
			if(features[0][i].features[0][j]==0){
				fprintf(stream,"%d,",0);
			}else{
				if(j<attributes->size()){
					fprintf(stream,"%4f,",features[0][i].features[0][j]);
				}else{
					fprintf(stream,"%d,",(DWORD)features[0][i].features[0][j]);
				}
			}
		}
		fprintf(stream,"%d,",(DWORD)features[0][i].mean_interval);
		fprintf(stream,"%d,",(DWORD)features[0][i].last_fatal);
		fprintf(stream,"%d,",features[0][i].lead_time);
		fprintf(stream,"%d,",features[0][i].start_date);
		fprintf(stream,"%4f,",features[0][i].location_pinpoint);
		fprintf(stream,"%4f,",features[0][i].location_recovery);
		fprintf(stream,"%d,",features[0][i].fatal_start_date);
		fprintf(stream,"%d\n",features[0][i].label);
	}
	fclose(stream);
}

std::vector<DWORD>* chars2ints(std::vector<char*>* inputs){
	std::vector<DWORD>* result=new std::vector<DWORD>(inputs->size());
	WORD i;
	for(i=0;i<(WORD)inputs->size();i++){
		result[0][i]=atoi(inputs[0][i]);
	}
	return result;
}
