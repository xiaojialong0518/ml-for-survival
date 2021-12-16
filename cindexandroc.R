#calculate and compare C-index
library(Hmisc)
library("survcomp")
library("survival")
library("prodlim")
library("survcomp")
cox <- read.csv("C:/code/rocdatacox.csv")
cox2 <- read.csv("C:/code/rocdatacox2.csv")
svm <- read.csv("C:/code/rocdatasvm.csv")
rsf <- read.csv("C:/code/rocdatarsf.csv")

options(digits=3)
#cox-en
ccox<- concordance.index(x=-cox$marker, surv.time=svm$time, surv.event=svm$os,method="noether")
ccox$c.index
ccox$lower
ccox$upper
#cox
ccox2 <- concordance.index(x=cox2$marker, surv.time=svm$time, surv.event=svm$os,method="noether")
ccox2$c.index
ccox2$lower
ccox2$upper
#svm
csvm<- concordance.index(x=svm$marker, surv.time=svm$time, surv.event=svm$os,method="noether")
csvm$c.index
csvm$lower
csvm$upper
#rsf
crsf<- concordance.index(x=rsf$marker, surv.time=svm$time, surv.event=svm$os,method="noether")
crsf$c.index
crsf$lower
crsf$upper

cindex.comp(crsf, ccox)
cindex.comp(crsf, ccox2)
cindex.comp(crsf, csvm)
cindex.comp(ccox, ccox2)
cindex.comp(ccox, csvm)
cindex.comp(ccox2, csvm)

#time-dependent ROC curves
library(survivalROC)
library(survcomp)
#library(readxl)
#cox is EN cox2 is traditional
cox <- read.csv("C:/Users/xiao/Desktop/研究生毕业论文/rocdatacox.csv")
coxr3=  survivalROC(Stime= cox$time, status=cox$os, marker=-cox$marker,  predict.time=36,method = "KM")
coxr5=  survivalROC(Stime= cox$time, status=cox$os, marker=-cox$marker,  predict.time=60,method = "KM")
coxr10=  survivalROC(Stime= cox$time, status=cox$os, marker=-cox$marker,  predict.time=120,method = "KM")

cox2 <- read.csv("C:/Users/xiao/Desktop/研究生毕业论文/rocdatacox2.csv")
cox2r3=  survivalROC(Stime= cox2$time, status=cox2$os, marker=cox2$marker,  predict.time=36,method = "KM")
cox2r5=  survivalROC(Stime= cox2$time, status=cox2$os, marker=cox2$marker,  predict.time=60,method = "KM")
cox2r10=  survivalROC(Stime= cox2$time, status=cox2$os, marker=cox2$marker,  predict.time=120,method = "KM")

svm <- read.csv("C:/Users/xiao/Desktop/研究生毕业论文/rocdatasvm.csv")
svmr3=  survivalROC(Stime= svm$time, status=svm$os, marker=svm$marker,  predict.time=36,method = "KM")
svmr5=  survivalROC(Stime= svm$time, status=svm$os, marker=svm$marker,  predict.time=60,method = "KM")
svmr10=  survivalROC(Stime= svm$time, status=svm$os, marker=svm$marker,  predict.time=120,method = "KM")

rsf <- read.csv("C:/Users/xiao/Desktop/研究生毕业论文/rocdatarsf.csv")
rsfr3=  survivalROC(Stime= rsf$time, status=rsf$os, marker=rsf$marker,  predict.time=36,method = "KM")
rsfr5=  survivalROC(Stime= rsf$time, status=rsf$os, marker=rsf$marker,  predict.time=60,method = "KM")
rsfr10=  survivalROC(Stime= rsf$time, status=rsf$os, marker=rsf$marker,  predict.time=120,method = "KM")


#calculate for D-index
a <- D.index(x=-cox$marker, surv.time=cox$time, surv.event=cox$os,method.test = c("logrank"))
b <- D.index(x=cox2$marker, surv.time=cox2$time, surv.event=cox2$os,method.test = c("logrank"))
c <- D.index(x=svm$marker, surv.time=svm$time, surv.event=svm$os,method.test = c("logrank"))
d <- D.index(x=rsf$marker, surv.time=rsf$time, surv.event=rsf$os,method.test = c("logrank"))

cox$subtype <- cut(-cox$marker,breaks =c(min(-cox$marker),median(-cox$marker),max(-cox$marker)),labels = c("low","high"))
cox2$subtype <- cut(cox2$marker,breaks =c(min(cox2$marker),median(cox2$marker),max(cox2$marker)),labels = c("low","high"))
svm$subtype <- cut(svm$marker,breaks =c(min(svm$marker),median(svm$marker),max(svm$marker)),labels = c("low","high"))
rsf$subtype <- cut(rsf$marker,breaks =c(min(rsf$marker),median(rsf$marker),max(rsf$marker)),labels = c("low","high"))

#output data to generate KM curves in python
write.csv(cox,"coxforkm.csv")
write.csv(cox2,"cox2forkm.csv")
write.csv(svm,"svmforkm.csv")
write.csv(rsf,"rsfforkm.csv")

#3 years
coxr=coxr3
cox2r=cox2r3
svmr=svmr3
rsfr=rsfr3
coxr3$AUC
cox2r3$AUC
svmr3$AUC
rsfr3$AUC
path= "C:/Users/xiao/Desktop/研究生毕业论文/ROC3.jpg"
tit="At 3 Years"
png(file = path,width=600*3,height=3*600,res=72*3)
#par(pin = c(2.5,2.5))
plot(cox2r$FP, cox2r$TP, 
     xlab=(""), ylab="",
     type="l",col="#1f77b4",xlim=c(0,1), ylim=c(0,1), lwd=1.8,font.axis=1,
     cex.axis=1,cex.main=1.5,
     main=tit)
par(new=TRUE)
plot(coxr$FP, coxr$TP, xlab=(""), ylab="", lwd=1.8,cex.axis=1,cex.main=1,font.axis=1,
     type="l",col="#ff7f0e",xlim=c(0,1), ylim=c(0,1))
par(new=TRUE)  
plot(svmr$FP, svmr$TP, xlab=(""), ylab="",lwd=1.8,cex.axis=1,cex.main=1,font.axis=1,
     type="l",col="#2ca02c",xlim=c(0,1), ylim=c(0,1))
par(new=TRUE)  
plot(rsfr$FP, rsfr$TP, xlab=(""), ylab="",lwd=1.8,cex.axis=1,cex.main=1,font.axis=1,
     type="l",col="#d62728",xlim=c(0,1), ylim=c(0,1))
abline(0,1,col="gray",lty=2)
#par(pin = c(3,3))
legend("bottomright",c("Cox","Cox-EN","SVM","RSF"),col=c("#1f77b4","#ff7f0e","#2ca02c","#d62728"),lty=1,lwd=1.8,cex=1.5)
dev.off()

#5 years
coxr=coxr5
cox2r=cox2r5
svmr=svmr5
rsfr=rsfr5
coxr5$AUC
cox2r5$AUC
svmr5$AUC
rsfr5$AUC
path= "C:/Users/xiao/Desktop/研究生毕业论文/ROC5.jpg"
tit="At 5 Years"

png(file = path,width=600*3,height=3*600,res=72*3)
#par(pin = c(2.5,2.5))
plot(cox2r$FP, cox2r$TP, 
     xlab=(""), ylab="",
     type="l",col="#1f77b4",xlim=c(0,1), ylim=c(0,1), lwd=1.8,font.axis=1,
     cex.axis=1,cex.main=1.5,
     main=tit)
par(new=TRUE)
plot(coxr$FP, coxr$TP, xlab=(""), ylab="", lwd=1.8,cex.axis=1,cex.main=1,font.axis=1,
     type="l",col="#ff7f0e",xlim=c(0,1), ylim=c(0,1))
par(new=TRUE)  
plot(svmr$FP, svmr$TP, xlab=(""), ylab="",lwd=1.8,cex.axis=1,cex.main=1,font.axis=1,
     type="l",col="#2ca02c",xlim=c(0,1), ylim=c(0,1))
par(new=TRUE)  
plot(rsfr$FP, rsfr$TP, xlab=(""), ylab="",lwd=1.8,cex.axis=1,cex.main=1,font.axis=1,
     type="l",col="#d62728",xlim=c(0,1), ylim=c(0,1))
abline(0,1,col="gray",lty=2)
#par(pin = c(3,3))
legend("bottomright",c("Cox","Cox-EN","SVM","RSF"),col=c("#1f77b4","#ff7f0e","#2ca02c","#d62728"),lty=1,lwd=1.8,cex=1.5)
dev.off()

#10 years
coxr=coxr10
cox2r=cox2r10
svmr=svmr10
rsfr=rsfr10
coxr10$AUC
cox2r10$AUC
svmr10$AUC
rsfr10$AUC
path= "C:/Users/xiao/Desktop/研究生毕业论文/ROC10.jpg"
tit="At 10 Years"

png(file = path,width=600*3,height=3*600,res=72*3)
#par(pin = c(2.5,2.5))
plot(cox2r$FP, cox2r$TP, 
     xlab=(""), ylab="",
     type="l",col="#1f77b4",xlim=c(0,1), ylim=c(0,1), lwd=1.8,font.axis=1,
     cex.axis=1,cex.main=1.5,
     main=tit)
par(new=TRUE)
plot(coxr$FP, coxr$TP, xlab=(""), ylab="", lwd=1.8,cex.axis=1,cex.main=1,font.axis=1,
     type="l",col="#ff7f0e",xlim=c(0,1), ylim=c(0,1))
par(new=TRUE)  
plot(svmr$FP, svmr$TP, xlab=(""), ylab="",lwd=1.8,cex.axis=1,cex.main=1,font.axis=1,
     type="l",col="#2ca02c",xlim=c(0,1), ylim=c(0,1))
par(new=TRUE)  
plot(rsfr$FP, rsfr$TP, xlab=(""), ylab="",lwd=1.8,cex.axis=1,cex.main=1,font.axis=1,
     type="l",col="#d62728",xlim=c(0,1), ylim=c(0,1))
abline(0,1,col="gray",lty=2)
#par(pin = c(3,3))
legend("bottomright",c("Cox","Cox-EN","SVM","RSF"),col=c("#1f77b4","#ff7f0e","#2ca02c","#d62728"),lty=1,lwd=1.8,cex=1.5)
dev.off()











