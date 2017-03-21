# Bayesian Multi-view Multi-task Linear Regression
# Copyright (R), all rights reserved by authors.
#
# Authors: 
# Suleiman Khan (khan.suleiman@gmail.com)
# Muhammad Ammad (myammad@gmail.com)

library(glmnet)
source("run.GLMNET.R")
source("run.nreg.R")

regress.cv <- function(x,y,method,standardize.x = TRUE,standardize.y=TRUE,opts=NULL,folds="LOOCV")
{
  if(folds == "LOOCV")
    folds = nrow(x)

  #else it is nFolds
  return(regress.nFolds(x,y,method,standardize.x,standardize.y,opts,nfolds=folds))
}


regress.nFolds <- function(x,y,method,standardize.x = TRUE,standardize.y=TRUE,opts=NULL,nfolds)
{
  if(nfolds <= 2) { print("nFolds should be 3 or more.. EXITING"); return(); }
  S = nrow(x)
  dX = ncol(x)
  dY = ncol(y)
  if(S == nfolds) { foldid = 1:S
  } else { finds = sample(S,S,replace=FALSE); foldid = ceiling(finds/(S/nfolds)); }
  ynew = matrix(NA,S,dY)
  res = list()

 for(fid in 1:nfolds)
 {
   samp = which(foldid == fid)
   if(length(y)==nrow(y))
     yN = matrix(y[-samp,],nrow(y)-1,1)
   else
     yN = y[-samp,]
   
   xN = x[-samp,]

   if(standardize.x)
     xN = ztransform(xN,1)
   else
     xN = list(mat = xN,cs=rep(1,dX),cm=rep(0,dX))
   ##xN = varNorm(x,1)
   
   if(standardize.y)
     yN = ztransform(yN,1)
   else
     yN = list(mat = yN,cs=rep(1,dY),cm=rep(0,dY))
   
   xTrain = xN$mat
   yTrain = yN$mat
   if(length(samp)==1) { #LOOCV
    xTest = matrix((x[samp,]- xN$cm)/xN$cs,1,ncol(x))
   } else {
     xTest = x[samp,]
     xTest = (xTest-matrix(xN$cm,nrow(xTest),dX,byrow=T))/matrix(xN$cs,nrow(xTest),dX,byrow=T)
   }
   inds = which(is.na(xTest)); if(length(inds)>0) xTest[inds] = 0;

   if(method == "GLMNET.Lasso" || method == "GLMNET.Lasso.Univariate")
   {
     ll <- runGLMNET(xTrain,yTrain,xTest,yN,INTERCEPT,type="Lasso"); fit = ll$fit; ynew[samp,] = ll$ynew.samp; rm(ll)
   }
   
   if(method == "GLMNET.EN" || method == "GLMNET.EN.Univariate")
   {
     ll <- runGLMNET(xTrain,yTrain,xTest,yN,INTERCEPT,type="EN"); fit = ll$fit; ynew[samp,] = ll$ynew.samp; rm(ll)
   }
   
   if(method == "MV.STAN")
   {
     dataFold = list(dX = dX, dXm=array(opts$dXm,dim=length(opts$dXm)),dY = ncol(Y), M=opts$M, X = xTrain,Y = yTrain,nY = nrow(xTrain))
     dataFold$tau_v=opts$tau_v; dataFold$lambda_v = opts$lambda_v; dataFold$wp=array(opts$wp,dim=length(opts$wp)); dataFold$taup=array(opts$taup,dim=length(opts$taup))
     fit <- runMVSTAN(data=dataFold,xTest,yN,INTERCEPT); ynew[samp,] = fit$ynew.samp;
   }
   
   fit$xTest = xTest; fit$yN = yN; 
   res[[fid]] = fit
   if(fid==nfolds){
     if(length(y)==nrow(y))
     {
       yc = round(cor(ynew,y),2); 
       yr = mean( round(sqrt(mean( (ynew - y)^2))/
                          sqrt(mean( (y - mean(y) )^2)),2) )
     }
     else
     {
       yc = round(mean(na.omit(diag(cor(ynew,y)))),2); 
       yr = mean( round(sqrt(colMeans( (ynew - y)^2))/
                          sqrt(colMeans( (y - matrix(colMeans(y),nrow(y),ncol(y),byrow=T) )^2)),2) )
     }
     cat(paste0("c",round(yc,2))); 
     cat(paste0(" r",round(yr,2)," ")); 
   }
   cat(paste(fid,method,"\n"))
   #print(proc.time()/3600)
 }
  if(is.na(yc)) yc = 0;
 list(res=res,yPred=ynew,yRMSE=yr,yCor = yc,opts=opts)
}

getBeta.CV <- function(method,run,featnames)
{
  PredBeta = 0
  if(length(grep(method,pattern="GLMNET")) >0)
  {
    if(ncol(run$yPred)>1) npb = nrow(run$res[[1]]$beta[[1]])
    else npb = nrow(run$res[[1]]$beta)
    
    PredBeta = matrix(0,ncol(run$yPred),npb)
    for(cvs in 1:length(run$res)){
      PredBeta = PredBeta + getBeta.GLMNET(fit=run$res[[cvs]],featnames,yCol=ncol(run$yPred))
    }
    PredBeta = PredBeta/length(run$res);
  }
  return(PredBeta)
}

getBeta <- function(method,fit,featnames)
{
  PredBeta = 0
  if(length(grep(method,pattern="GLMNET")) >0)
  {
    PredBeta = getBeta.GLMNET(fit,featnames)
  }
  if(length(grep(method,pattern="MV.STAN")) >0)
  {
    PredBeta = getBeta.MV.STAN(fit)
  }
  return(PredBeta)
}

