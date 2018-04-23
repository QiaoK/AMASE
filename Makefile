CC=g++
CFLAGS= -O2 -Wall -Wextra -c
LIBS = -lm
ST_CLUSTERING_OBJS = st_clustering.o random.o data_functions.o st_clustering_main.o
FATAL_FEATURES_OBJS = st_clustering.o random.o data_functions.o feature_save_main.o
All: st_clustering fatal_features
st_clustering : $(ST_CLUSTERING_OBJS)
	$(CC) -o $@ $(ST_CLUSTERING_OBJS) $(LIBS)
fatal_features : $(FATAL_FEATURES_OBJS)
	$(CC) -o $@ $(FATAL_FEATURES_OBJS) $(LIBS)
%.o: %.cpp
	$(CC) $(CFLAGS) $<  
%.o: %.c
	$(CC) $(CFLAGS) $<
clean:
	rm -rf *.o
	rm -rf st_clustering_test
