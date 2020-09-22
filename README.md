This software implements the anomaly detection algorithms proposed in paper https://ieeexplore.ieee.org/abstract/document/9006046.
To reproduce the results, you need to go to ANL ALCF website to download MIRA data first https://reports.alcf.anl.gov/data/mira.html. 

# System requirement

- gcc compiler that supports C99.
- python3
- pytorch 1.0
- R

I suggest users to install these packages with anaconda.

# Compile

Use make file to compile code and test cases with command "make".

# Executables
Unzip all RAS_EVENT data you have downloaded. Unzip the dataset and put in the same folder.

* Use the R code in https://github.com/QiaoK/AMASE/blob/master/developer_note.txt to clean up the data.
* Compile the C++ code. Use the "./streaming_features" to generate the training and testing features.
* Usage: ./streaming_feature warn_table_name fatal_table_name fatal_t_threshold window_size lead_time_count
* Example: ./streaming_feature sys_events.csv sys_fatal_spatial.csv 1 300 4
* Run the python training code for neural network training.
* Usage: python training_lead_time.py feature_file_name training_epochs retrain(0 or 1) reload_data(0 or 1) lead_time_interval(3600) lead_time_stride(1)
* feature_file_name is the output file generated by ./streaming_feature

# Contact

Contact me at qiao.kang@eecs.northwestern.edu if you have questions.
