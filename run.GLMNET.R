# Bayesian Multi-view Multi-task Linear Regression
# Copyright (R), all rights reserved by authors.
#
# Authors: 
# Suleiman Khan (suleiman.khan@helsinki.fi)
# Muhammad Ammad (muhammad.ammad-ud-din@helsinki.fi )

source("HelpingFunctions.R")
require("glmnet")

getBeta.GLMNET <- function(fit,featnames,yCol)
{
  PredBeta = 0
  cf = coef(fit,s= fit$ld)
  if(yCol>1){
    PredBeta = matrix(0,length(cf),length(cf[[1]]))
    for(icf in 1:length(cf)){
      PredBeta[icf,] = PredBeta[icf,] + t(as.matrix(cf[[icf]]))
    }
  } else {
    PredBeta = t(as.matrix(cf))
  }
  if(nrow(PredBeta)>1) { colnames(PredBeta) = c("intercept",featnames);
  } else { names(PredBeta) = c("intercept",featnames);}
  if(INTERCEPT == FALSE) PredBeta = PredBeta[,-1];
  return(PredBeta)
}

runGLMNET <- function(xTrain,yTrain,xTest,yN,INTERCEPT,type="Lasso")
{  
  ## We can do without CV - but need one Lambda for Prediction
  ## More details on how to run and what each type is: http://cran.r-project.org/web/packages/glmnet/vignettes/glmnet_beta.html
  # model = glmnet(xTrain,yTrain,family="mgaussian",alpha=1,standardize.response=FALSE,standardize=FALSE,intercept=FALSE)  
  # predict(model, newx = xTest)    
  if(length(yTrain)==nrow(yTrain))
    family="gaussian"
  else
    family="mgaussian"
  
  internal.nfolds=floor(nrow(xTrain)/2)
  if(internal.nfolds > 10) internal.nfolds = 10 #default
  finds = sample(nrow(xTrain),nrow(xTrain),replace=FALSE); foldid = ceiling(finds/(nrow(xTrain)/internal.nfolds))
  
  if(type=="Lasso")
  {
    alpha = 1
    fit = try(cv.glmnet(xTrain,yTrain,nfolds=internal.nfolds,type.measure="mse",family=family,alpha=alpha,standardize.response=!INTERCEPT,standardize=TRUE,intercept=INTERCEPT))
    #sqrt(res.glmnet$cvm[which(res.glmnet$lambda == res.glmnet$lambda.1se)])  
    #coef(cvfit, s = "lambda.1se") #lambda.1se lambda.min
    #TODO use 1/N variance formula
  } 
  if(type=="Ridge")
  {
    alpha = 0
    fit = try(cv.glmnet(xTrain,yTrain,nfolds=internal.nfolds,type.measure="mse",family=family,alpha=alpha,standardize.response=!INTERCEPT,standardize=TRUE,intercept=INTERCEPT))
  }
  if(type=="EN")
  {
    alphas = 1:10/10
    fitlist = list()
    for(a.cv in 1:length(alphas))
    {
      fitlist[[a.cv]] = try(cv.glmnet(xTrain,yTrain,nfolds=internal.nfolds,foldid=foldid,type.measure="mse",family=family,alpha=alphas[a.cv],standardize.response=!INTERCEPT,standardize=TRUE,intercept=INTERCEPT))
    }
    si = which.min(lapply(fitlist,function(x) {x$lambda.1se}))
    fit = fitlist[[si]]; alpha=alphas[si]
  }
  
  if(class(fit)=="try-error")
  {
    fit = glmnet(xTrain,yTrain,family=family,alpha=alpha,standardize.response=!INTERCEPT,standardize=FALSE,intercept=INTERCEPT)
    ld = fit$lambda[which(fit$dev[2:length(fit$dev)] - fit$dev[1:(length(fit$dev)-1)] < 0.01)[1]]
  } 
  else 
  {
    ld = fit$lambda.1se
    fit = glmnet(xTrain,yTrain,family=family,alpha=alpha,standardize.response=!INTERCEPT,standardize=TRUE,intercept=INTERCEPT)
    fit$ld = ld
  }
  
  ynew.samp = 0;
  if(length(xTest)>0) {
    ynew.samp = predict(fit, newx = xTest, s = ld)
    if(nrow(xTest)==1) {
      ynew.samp = ynew.samp*yN$cs + yN$cm;
    } else {
      ynew.samp = ynew.samp[,,1]*matrix(yN$cs,nrow(ynew.samp),ncol(ynew.samp),byrow=T) + matrix(yN$cm,nrow(ynew.samp),ncol(ynew.samp),byrow=T);
    }
  }
  if(DEBUG == TRUE) plot(fit)
  
  return(list(fit=fit,ynew.samp=ynew.samp))
}