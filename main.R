# Bayesian Multi-view Multi-task Linear Regression
# Copyright (R), all rights reserved by authors.
#
# Authors: 
# Suleiman Khan (suleiman.khan@helsinki.fi)
# Muhammad Ammad (muhammad.ammad-ud-din@helsinki.fi )

rm(list=ls())

library(rstan)
#rstan_options(auto_write = TRUE)
#options(mc.cores = parallel::detectCores())
#set_cppo(mode = "fast")
#library(ggmcmc)
set.seed(101)

source("toydata.R")
source("run.nreg.R")
source("regress.CV.R")

nExps = 1
M <- 1
nY = 20;
pt = paste0("res.",M,".",nY)
tabResult = matrix(NA,nExps,4); colnames(tabResult) = c("glmnet.yRMSE","glmnet.yCor","stan.yRMSE","stan.yCor")
for(iter in 1:nExps)
{
  #Generate Data
  dXm = sample(40:50,M,replace=T); dX = sum(dXm); dY = 6; K = 1 #
  dataList = generateVisualDataSingleK(N=nY,M=M,P=dXm,D=dY,StandardizeY="none");
  X = dataList$X;Y = dataList$Y; H = dataList$H; W = dataList$W; noise = dataList$noise; Beta = dataList$Beta;
  rcX = rowConcat(X)
  
  print("Running GLMNET..."); 
  INTERCEPT = FALSE; DEBUG = FALSE;
  run.glmnet = regress.cv(x = rcX, y = Y,method="GLMNET.EN",standardize.x = FALSE,standardize.y=FALSE,opts=NULL,folds="LOOCV")
  GLMNET.EN.beta = getBeta.CV(method="GLMNET.EN",run.glmnet,featnames=colnames(rcX))

  print("Running MVSTAN..."); 
  INTERCEPT = FALSE; DEBUG = FALSE;
  opts = list(lambda_v = 1e-1,tau_v=1,wp=rep(1,dY),taup=rep(1,M),dXm=dXm,M=M);
  run.stan = regress.cv(x = rcX, y = Y, method="MV.STAN",standardize.x = FALSE,standardize.y=FALSE,opts=opts,folds="LOOCV")
  post = getPosteriorEV.MV.STAN.CV(run.stan)
  
  print("result")
  print(run.glmnet$yRMSE)
  print(run.glmnet$yCor)
  print(run.stan$yRMSE)
  print(run.stan$yCor)
  print(iter)
  tabResult[iter,] = round(c(run.glmnet$yRMSE,run.glmnet$yCor,run.stan$yRMSE,run.stan$yCor),3)

 #plot single run
 if(iter == 1) {
 pdf(file=paste0(pt,".pdf"),width=7,height=5)
 GLMbeta = apply(GLMNET.EN.beta,2,mean)
 limDat = c(GLMbeta,as.vector(unlist(Beta)),post$beta)
 GLMbeta = GLMbeta*max(abs(post$beta))/max(abs(GLMbeta)) #conform for plotting
 plot(1:ncol(post$beta),post$beta[1,],col=palette()[1],pch=16,ylim=c(min(limDat),max(limDat)));
 points(GLMbeta,col=palette()[2],pch=17,cex=0.8)
 ind = 0;
 for(m in 1:M) {
     points(1:nrow(Beta[[m]])+ind,Beta[[m]][,1],col=palette()[m+2],type="l",lwd=2);
     ind = ind + nrow(Beta[[m]]);
   }
 legend("topright",legend=c("MVLR","Elastic Net",paste0("M",1:M," TRUE")),col=palette()[1:(M+2)],pch=c(16,17,rep(15,M)))
 plot(1:ncol(post$W),post$W[1,],col=palette()[2],pch=16,ylim=c(0,max(W,post$W)+0.1),cex=1.5); 
 points(1:ncol(W),W[1,],col=palette()[3],pch=17)
 legend("topright",legend=c("Learned","Original"),col=palette()[2:3],pch=c(16,17))
 dev.off()
 }
}
