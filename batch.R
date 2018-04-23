min_time<-function(x,Y){
	t<-Y-x
	t<--t[t<=0]
	if(length(t)==0){
		result<--1
	}else{
		result<-min(t)
	}
	as.integer(result)
}
shift_compare<-function(r,name){
	k<-r[,name]
	k<-k[2:length(k)]==k[1:(length(k)-1)]
	k
}

shift_difference<-function(r,name,diff){
	k<-r[,name]
	k<-k[(1+diff):length(k)]-k[1:(length(k)-diff)]
	k
}

create_event<-function(subsequence,attributes,r){
	subsequence<-unlist(subsequence)
	result<-rep(0:0,length(attributes)+2)
	names(result)<-c(attributes,"START","END")
	r<-r[subsequence,]
	v<-paste(r[,"SEVERITY"],r[,"COMPONENT"],r[,"CATEGORY"])
	t<-table(v)
	s<-sum(t)
	result[names(t)]<-t/s
	result["START"]<-min(r[,"EVENT_TIME"])
	result["END"]<-max(r[,"EVENT_TIME"])
	result
}

filter_time<-function(r,interval,filter,attributes){
	size<-nrow(r)
	t<-r[,"EVENT_TIME"]
	t<-t[2:length(t)]-t[1:(length(t)-1)]
	if(filter==0){
		start<-c(0,which(t>interval))
		if(length(start)>1){
			end<-c(start[2:length(start)],size)
		}else{
			end<-size
		}
		start<-start+1
		subsequences<-mapply(seq,start,end,MoreArgs=list(by=1))
		l<-lapply(subsequences,create_event,attributes=attributes,r=r)
		result <- data.frame(matrix(unlist(l), ncol=(length(attributes)+2), byrow=T))
		colnames(result) <- c(attributes,"START","END")
		result <- result[,colSums(result!=0)>0]
		result
	}else if(filter==1){
		block<-shift_compare(r,"BLOCK")
		removed<-intersect(which(t<=interval),which(block==TRUE))	
	}else if(filter==2){
		component<-shift_compare(r,"COMPONENT")
		category<-shift_compare(r,"CATEGORY")
		removed<-intersect(which(t<=interval),intersect(which(component==TRUE),which(category==TRUE)))
	}else if(filter==3){
		component<-shift_compare(r,"COMPONENT")
		category<-shift_compare(r,"CATEGORY")
		block<-shift_compare(r,"BLOCK")
		removed<-intersect(intersect(which(t<=interval),intersect(which(component==TRUE),which(category==TRUE))),which(block==TRUE))
	}
}

temporal_clustering<-function(r,interval){
	size<-nrow(r)
	t<-r[,"EVENT_TIME"]
	t<-t[2:length(t)]-t[1:(length(t)-1)]
	start<-c(0,which(t>interval))
	if(length(start)>1){
		end<-c(start[2:length(start)],size)
	}else{
		end<-size
	}
	start<-start+1
	result<-data.frame(start,end)
	colnames(result)<-c("START","END")
	result
}

map_index<-function(index,t_cluster,cluster_index){
	list(cluster_index[t_cluster[index,"START"]:t_cluster[index,"END"]])
}

split_spatial_cluster<-function(cluster_index,res,interval){
	if(length(cluster_index)>1){
		cluster_index<-unlist(cluster_index)
		t_cluster<-temporal_clustering(res[cluster_index,],interval)
		x<-vapply(seq(1,nrow(t_cluster),by=1),map_index,t_cluster=t_cluster,cluster_index=cluster_index,FUN.VALUE=list(1),USE.NAMES=FALSE)
		list(x)
	}else{
		list(cluster_index)
	}
}

cluster_filter<-function(index,clusters,res,interval){
	start<-clusters[index,"START"]
	end<-clusters[index,"END"]
	x<-""
	print(c(start,end))
	locations<-rep("123",end-start+1)
	vapply(seq(start,end,by=1),function(index,locations,res_locations,start){locations[index-start]<<-unlist(strsplit(toString(res_locations[index]),"-"))[1]},locations=locations,res_locations=res[,"LOCATION"],start=start-1,FUN.VALUE=x,USE.NAMES=FALSE)
	locations.unique<-unique(locations)
	x<-rep(list(1),length(locations.unique))
	vapply(seq(1,length(locations.unique)),function(index,x,locations.unique,locations,start){x[index]<<-list(start+which(locations==locations.unique[index]))},x=x,locations=locations,locations.unique=locations.unique,start=start-1,FUN.VALUE=list(1),USE.NAMES=FALSE)
	result<-lapply(x,split_spatial_cluster,res=res,interval=interval)
	list(result)
}

spatial_filtering<-function(clusters,res,interval){
	result<-vapply(seq(1,nrow(clusters),by=1),cluster_filter,clusters=clusters,res=res,FUN.VALUE=list(1),interval)
	result
}

#sys_fatal_spatial<-subset(sys_fatal,LOCATION!="")
#t_clusters<-temporal_clustering(sys_fatal_spatial,reverse_cdf(sys_fatal_spatial,.999))
#s_clusters<-spatial_filtering(t_clusters,sys_fatal_spatial)
#save.image()

parse_location<-function(location){
	x<-unlist(strsplit(toString(location),"-"))[1]
	racks[[toString(location)]]<<-x
	x
}

parse_locations<-function(locations){
	x<-"1"
	vapply(locations,parse_location,FUN.VALUE=x,USE.NAMES=FALSE)
	x
	#for(i in 1:length(locations)){
	#	result[i]<-unlist(strsplit(toString(locations[i]),'-'))[1]
	#}}
}

match_fatal<-function(compressed_warn,compressed_fatal){
	compressed_warn<-unlist(compressed_warn)
	start<-compressed_warn["START"]
	end<-compressed_warn["END"]
	result<-which(compressed_fatal[,"START"]<=end)
	result<-intersect(result,which(compressed_fatal[,"START"]>=start))
	result<-intersect(result,which(compressed_fatal[,"END"]>=end))
	if(length(result)==0){
		result<--1
	}
	result[1]
}

create_features<-function(compressed_warn,compressed_fatal){
	compressed_warn.list<-split(compressed_warn, seq(nrow(compressed_warn)))
	feature_index<-lapply(compressed_warn.list,match_fatal,compressed_fatal=compressed_fatal)
	feature_index<-unlist(feature_index)
	compressed_warn<-cbind(compressed_warn,feature_index)
	colnames(compressed_warn)[ncol(compressed_warn)]<-"FATAL"
	compressed_warn[,"FATAL"]<-replace(compressed_warn[,"FATAL"],compressed_warn[,"FATAL"]==-1,"0")
	compressed_warn[,"FATAL"]<-replace(compressed_warn[,"FATAL"],compressed_warn[,"FATAL"]>0,"1")
	compressed_warn[,"FATAL"]<-gsub("0","NON-FATAL",as.factor(compressed_warn[,"FATAL"]))
	compressed_warn[,"FATAL"]<-gsub("1","FATAL",as.factor(compressed_warn[,"FATAL"]))
	compressed_warn
}

adjacent_fatal<-function(compressed_warn,compressed_fatal){
	compressed_warn<-unlist(compressed_warn)
	start<-compressed_warn["START"]
	end<-compressed_warn["END"]
	result<-which(compressed_fatal[,"START"]<=end)
	result<-intersect(result,which(compressed_fatal[,"START"]>=start))
	result<-intersect(result,which(compressed_fatal[,"END"]>=end))
	if(length(result)==0){
		result<-c(-1,-1,-1)
	}else{
		result<-c(compressed_warn["START"],compressed_warn["END"],compressed_fatal[result[1],"START"])
	}
	result
}

adjacent_feature<-function(compressed_warn,compressed_fatal){
	compressed_warn.list<-split(compressed_warn, seq(nrow(compressed_warn)))
	feature_index<-lapply(compressed_warn.list,adjacent_fatal,compressed_fatal=compressed_fatal)
	result <- data.frame(matrix(unlist(feature_index), ncol=3, byrow=T))
	colnames(result) <- c("START","END","FATAL_END")
	result
}

reverse_cdf<-function(r,quantile){
	t<-r[,"EVENT_TIME"]
	t<-sort(t[2:length(t)]-t[1:(length(t)-1)])
	t[as.integer(length(t)*quantile)]
}

compute_cdf<-function(r,bound){
	t<-r[,"EVENT_TIME"]
	t<-sort(t[2:length(t)]-t[1:(length(t)-1)])
	length(t[t>=bound])/length(t)
}

set_hashtable<-function(control,cobalt){
	#print(control[index])
	#print(cobalt[index])
	#print(table[index,"CONTROL"])
	#print(table[index,"COBALT"])
	ht[[toString(control)]]<<-cobalt
}

control_to_cobalt<-function(control){
	ht[[toString(control)]]
}

create_job_fatal_matrix<-function(th,res2,cobalts){
	feature<-paste(paste(res2[,"SEVERITY"],paste(res2[,"COMPONENT"],res2[,"CATEGORY"],sep=" "),sep=" "))
	v<-table(data.frame(cobalts,feature))
	v
}

construct_nominal_table<-function(labels){
	labels<-unique(labels)
	normalizeds<-seq(1,length(labels),by=1)/length(labels)
	mapply(function(label,normalized){ht[[toString(label)]]<<-normalized},label=labels,normalized=normalizeds)
}

normalize_numeric<-function(column,djc){
	a<-djc[,column]
	min_a<-min(a)
	a<-(a-min_a)/(max(a)-min_a)
	a
}

normalize_nominal<-function(column,djc){
	x<-djc[,column]
	a<-construct_nominal_table(x)
	a<-1.0
	result<-vapply(x,function(label){ht[[toString(label)]]},FUN.VALUE=a)
	result
}

create_job_feature<-function(djc){
	#nominals<-c('MODE','DATA_LOAD_STATUS','CAPABILITY','SIZE_BUCKETS3','PERCENTILE')
	numeric_features<-c('NODES_REQUESTED','CORES_REQUESTED',"REQUESTED_CORE_HOURS")
	nominal_features<-"PROJECT_NAME_GENID"
	numerics<-vapply(numeric_features,normalize_numeric,djc=djc,FUN.VALUE=as.numeric(djc[,1]))
	nominals<-vapply(nominal_features,normalize_nominal,djc=djc,FUN.VALUE=as.numeric(djc[,1]))
	result<-cbind(numerics,nominals,djc[,"EXIT_CODE"])
	#result<-cbind(nominals,djc[,"EXIT_CODE"])
	colnames(result)<-c(numeric_features,nominal_features,"EXIT_STATUS")
	#colnames(result)<-c(nominal_features,"EXIT_STATUS")
	rownames(result)<-djc[,"COBALT_JOBID"]
	result
}


