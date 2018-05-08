setwd("C:/cygwin64/home/Qia/AMASE")
lead_times<-read.table("lead_times.txt",head=F,sep=",")
v1<-as.numeric(sub(-1,0,lead_times$V1))
v2<-as.numeric(sub(-1,0,lead_times$V2))

v<-rep(1:1,nrow(lead_times))
for(i in 1:nrow(lead_times)){
	v[i]<-max(v1[i],v2[i])
}
hist(v,breaks=50)

plot_cdf<-function(v,stripe,xlim){
	it<-seq(1,xlim,stripe)
	v_cdf<-rep(1:1,length(it))
	for(i in 1:length(it)){
		index<-it[i]
		v_cdf[i]<-length(v[v<=index])/length(v)
	}
	pdf("lead_time1.pdf")
	plot(it,v_cdf,xlab="seconds",ylab="cumulative probability",cex.axis=1.4,cex.lab=1.5,main="Lead Time cdf")
	response<-log(1/(1-v_cdf))
	regression<-lm(response~0+it)
	lines(it,1-1/exp(regression$fitted.values))
	print(regression$coefficients)
	dev.off()
}