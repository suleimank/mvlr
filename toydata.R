# Bayesian Multi-view Multi-task Linear Regression
# Copyright (R), all rights reserved by authors.
#
# Authors: 
# Suleiman Khan (khan.suleiman@gmail.com)
# Muhammad Ammad (myammad@gmail.com)

library("mvtnorm")
source("HelpingFunctions.R")
generateData <- function(N,M,P,D,K,Hfunc,Wfunc,Bfunc,Pos=FALSE,StandardizeY="varNorm")
{
  print("-------------------------------------------------------------------------")

  print("Generating Data.......")
  noise = round(runif(D,0.01,0.5),2)
  H <- get(Hfunc)(M,K)
  W <- get(Wfunc)(D,K)
  X = list()
  Beta = list()
  Y = 0
  for(m in 1:M)
  {
    if(FALSE == Pos)
    {
      X[[m]] <- rmvnorm(N,mean=rep(0,P[m]),sigma=diag(1,P[m])); #matrix(rnorm(N*P[m]),N,P[m])
      X[[m]] = ztransform(X[[m]],1)$mat
    }
    else
    {
      X[[m]] <- abs(rmvnorm(N,mean=rep(0,P[m]),sigma=diag(1,P[m]))); #matrix(abs(rnorm(N*P[m])),N,P[m])
      X[[m]] = varNorm(X[[m]],1)$mat
    }
    colnames(X[[m]]) = paste0("M",m,".F",1:ncol(X[[m]]))
    Beta[[m]] <- matrix(0,P[m],K)
    for(k in which(H[m,] == 1))
    {
      Beta[[m]][,k] = get(Bfunc)(Beta[[m]][,k],m,k)
    }
    
    Y = Y + (X[[m]] %*% Beta[[m]] %*% W)
  }
  print(paste0("Data: M",m,", K",K,", N",N,", D",D,", P",sum(P),", avgN/S",round(mean(noise),2),sep=""))
  for(m in 1:M)
    print(paste0("Contribution (NMS) of View",m,": ",(mean((X[[m]] %*% Beta[[m]][,1] %*% W[1,])^2))/(mean(Y^2))))
  
  if(D ==1){
    Y = Y + rnorm(N)*noise*sd(Y)
  } else{
    Y = Y + rmvnorm(N,mean=rep(0,D),sigma=diag(noise*apply(Y,2,sd))^2)
  }
  colnames(Y) = paste0("D",1:ncol(Y))
  
  if(StandardizeY != "none")
  {
    dtmp = get(StandardizeY)(Y,1); Y = dtmp$mat;
    W = W/matrix(dtmp$cs,nrow(W),ncol(W),byrow=T)
  }
  return(list(Y=Y,X=X,noise=noise,H=H,W=W,Beta=Beta))
}

W1 <- function(D,K=1){
  W <- matrix(0,1,D); rownames(W) = paste0("C",1:1); colnames(W) = paste0("D",1:D)
  W[1,] = runif(D,0.7,1)
  W = W/sum(W)
  return(W)
}
B1 <- function(beta,m,k=1)
{
  val = 0.5
  nFeat = floor(length(beta)*0.25)
  if(length(beta)>10) nFeat = floor(length(beta)*0.1);
  beta[(1:nFeat)] = sample(c(-1,1),nFeat,replace=TRUE)*runif(nFeat,0.3,0.7)
  return(beta)
}
H1 <- function(M,K=1){
  H<- matrix(0,M,1); colnames(H) = paste0("C",1:1); rownames(H) = paste0("V",1:M);
  tk = ceiling(M/2); if(tk>5) tk = 5;
  H[1:tk,1] = 1;
  return(H)
}
visualW <- function(D,K){
  nItems = floor(D/K)
  W <- matrix(0,K,D); rownames(W) = paste0("C",1:K); colnames(W) = paste0("D",1:D)
  for(k in 1:K) W[k,1:nItems+(k-1)*nItems] = 1#runif(nItems,0.9,1)
  if(D >1) W = W/matrix(apply(abs(W),1,max),K,D,byrow=F);
  if(D==1) W <- matrix(1,K,D);
  return(W)
}
visualH <- function(M,K){
  H<- matrix(0,M,K); colnames(H) = paste0("C",1:K); rownames(H) = paste0("V",1:M); 
  H[,1] = 1; 
  if(K>1) H[1,2] = 1; 
  if(K>2 && M>1) H[2,3] = 1;
  return(H)
}
visualB <- function(beta,m,k)
{
  val = 0.25
  nFeat = floor(length(beta)/4)
  if(length(beta)>10) nFeat = floor(length(beta)/10);
  if(k == 1)
    beta[(1:nFeat)] = val
  else
    beta[(1:nFeat)+(k-1)*nFeat] = val
  return(beta)
}
generateVisualData <- function(N,M,P,D,K,StandardizeY="varNorm")
{
  dataList = generateData(N,M,P,D,K,"visualH","visualW","visualB",Pos=FALSE,StandardizeY);
  return(dataList)
}
generateVisualDataSingleK <- function(N,M,P,D,StandardizeY="varNorm")
{
  dataList = generateData(N,M,P,D,K=1,"H1","W1","B1",Pos=FALSE,StandardizeY);
  return(dataList)
}
generatePositiveData <- function(N,M,P,D,K,StandardizeY="varNorm")
{
  dataList = generateData(N,M,P,D,K,"visualH","visualW","visualB",Pos=TRUE,StandardizeY);
  return(dataList)
}

