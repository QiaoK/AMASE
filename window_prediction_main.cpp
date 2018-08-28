#include "data_functions.h"

int main(int argc,char** argv){
	DWORD period,observe_size,break_down;
	if(argc!=6){
		printf("Usage:./st_clustering filename1 filename2 period observe_size break_down\n");
		return 1;
	}
	period=atoi(argv[3]);
	observe_size=atoi(argv[4]);
	break_down=atoi(argv[5]);
	std::vector<std::vector<char*>*>* warn_table=read_table(argv[1]);
	std::vector<std::vector<char*>*>* fatal_table=read_table(argv[2]);
	std::vector<DWORD>* warn_dates=chars2ints(warn_table[0][COL_DATE]);
	std::vector<DWORD>* fatal_dates=chars2ints(fatal_table[0][COL_DATE]);
	printf("Data read finished\n");
	std::vector<Feature>* result=period_based_features(warn_table,warn_dates,fatal_dates,period,observe_size,break_down);

	char result_file_name[2000],header[200];
	sprintf(result_file_name,"window_features_%d_%d_%d.csv",period,observe_size,break_down);
	std::vector<std::string> *attributes=new std::vector<std::string>;
	unsigned int i;
	for(i=0;i<result[0][0].features->size();i++){
		sprintf(header,"%d",i);
		attributes->push_back(std::string(header));
	}
	write_features(result,result_file_name,attributes,10);
	printf("Feature write to local file finished\n");
	delete warn_dates;
	delete fatal_dates;
	delete_table(warn_table);
	delete_table(fatal_table);
	return 0;
}
