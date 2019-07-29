CC=g++
CFLAGS= -O2 -Wall -Wextra -c
LIBS = -lm
ST_CLUSTERING_OBJS = st_clustering.o random.o data_functions.o st_clustering_main.o
WINDOW_PREDICTION_OBJS = window_prediction.o st_clustering.o random.o data_functions.o window_prediction_main.o
FATAL_FEATURES_OBJS = st_clustering.o random.o data_functions.o feature_save_main.o
STREAMING_FEATURES_OBJS = st_clustering.o random.o data_functions.o streaming_features_main.o
All: st_clustering fatal_features window_prediction streaming_features
st_clustering : $(ST_CLUSTERING_OBJS) amase.h
	$(CC) -o $@ $(ST_CLUSTERING_OBJS) $(LIBS)
fatal_features : $(FATAL_FEATURES_OBJS)
	$(CC) -o $@ $(FATAL_FEATURES_OBJS) $(LIBS)
window_prediction : $(WINDOW_PREDICTION_OBJS) amase.h
	$(CC) -o $@ $(WINDOW_PREDICTION_OBJS) $(LIBS)
streaming_features : $(STREAMING_FEATURES_OBJS) amase.h
	$(CC) -o $@ $(STREAMING_FEATURES_OBJS) $(LIBS)
%.o: %.cpp
	$(CC) $(CFLAGS) $<  
%.o: %.c
	$(CC) $(CFLAGS) $<
clean:
	rm -rf *.o
	rm -rf st_clustering_test
	rm -rf streaming_features
	rm -rf window_prediction
