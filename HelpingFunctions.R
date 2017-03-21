# Bayesian Multi-view Multi-task Linear Regression
# Copyright (R), all rights reserved by authors.
#
# Authors: 
# Suleiman Khan (suleiman.khan@helsinki.fi)
# Muhammad Ammad (muhammad.ammad-ud-din@helsinki.fi )


#####################################
###### Sorting Functions ############
#####################################
my.sort <- function(x,d) { sort(x,decreasing=d) }

#####################################
###### NA Functions #################
#####################################
mean.na <- function(x) { mean(na.omit(x)) }
sd.na <- function(x) { sd(na.omit(x)) }
min.na <- function(x) { min(na.omit(x)) }
max.na <- function(x) { max(na.omit(x)) }
sum.na <- function(x) { sum(na.omit(x)) }
which.na <- function(x) { which(na.omit(x)) }
length.na <- function(x) { sum(!is.na(x)) }
sort.na.dec <- function(x) { my.sort(na.omit(x),d=TRUE) }
sort.na.inc <- function(x) { my.sort(na.omit(x),d=FALSE) }
na.omit.print <- function(x) { a = na.omit(x); print(a[1:nrow(a),]) }


#####################################
###### Normalization Functions ######
#####################################
ztransform <- function(mat, FFSC,verbose=FALSE){
  cmat = ncol(mat)
  rmat = nrow(mat)
  if(cmat==1)
  {
  cm = apply(mat,2,mean.na)
  if(verbose) print(paste("Number of Z Normalized Features(COLUMNS) is:",length(cm)));
  if(sum(is.na(mat)) >= (length(mat)-1) ) cm = 0
  mat = mat - cm
  cs = apply(mat,2,sd.na)
  cs[cs == 0] = 1
  mat = mat/cs
  return(list(mat=mat,cm=cm,cs=cs))
  }
  cm = apply(mat[,FFSC:cmat],2,mean.na)
  lm = apply(mat[,FFSC:cmat],2,length.na)
  inds = which(lm==1)
  if(length(inds)>0)
	cm[inds] = 0
  if(verbose) print(paste("Number of Z Normalized Features(COLUMNS) is:",length(cm)));
  mat[,FFSC:cmat] = mat[,FFSC:cmat] - matrix(cm,rmat,(cmat-FFSC+1),byrow=TRUE)
  cs = apply(mat[,FFSC:cmat],2,sd.na)
  cs[cs == 0] = 1
  inds = which(is.na(cs))
  if(length(inds)>0)
	cs[inds] = 1
  mat[,FFSC:cmat] = mat[,FFSC:cmat]/matrix(cs,rmat,(cmat-FFSC+1),byrow=TRUE)
  return(list(mat=mat,cm=cm,cs=cs))
}

meanNorm <- function(mat, FFSC,verbose=FALSE){
  cmat = ncol(mat)
  rmat = nrow(mat)
  if(cmat==1)
  {
    cm = apply(mat,2,mean.na)
    if(verbose) print(paste("Number of Z Normalized Features(COLUMNS) is:",length(cm)));
    if(sum(is.na(mat)) >= (length(mat)-1) ) cm = 0
    mat = mat - cm
    return(list(mat=mat,cm=cm))
  }
  cm = apply(mat[,FFSC:cmat],2,mean.na)
  lm = apply(mat[,FFSC:cmat],2,length.na)
  inds = which(lm==1)
  if(length(inds)>0)
    cm[inds] = 0
  if(verbose) print(paste("Number of Z Normalized Features(COLUMNS) is:",length(cm)));
  mat[,FFSC:cmat] = mat[,FFSC:cmat] - matrix(cm,rmat,(cmat-FFSC+1),byrow=TRUE)  
  return(list(mat=mat,cm=cm))
}

#
# 19.04.2016: Updated varNorm form correct $cm. 
#
varNorm <- function(mat, FFSC){
  cmat = ncol(mat)
  rmat = nrow(mat)
  if(cmat==1)
  {
  cs = apply(mat,2,sd.na)
  cs[cs == 0] = 1
  mat = mat/cs
  return(list(mat=mat,cs=cs,cm=0))
  }
  lm = apply(mat[,FFSC:cmat],2,length.na)
  inds = which(lm==1)
  cs = apply(mat[,FFSC:cmat],2,sd.na)
  cs[cs == 0] = 1
  inds = which(is.na(cs))
  if(length(inds)>0)
	cs[inds] = 1
  mat[,FFSC:cmat] = mat[,FFSC:cmat]/matrix(cs,rmat,(cmat-FFSC+1),byrow=TRUE)
  return(list(mat=mat,cs=cs,cm=0))
}

#
# Quantile Normalization for columns of df
#
quantile_normalisation <- function(df){
  
  df = as.matrix(df)
  missing.inds = which(is.na(df))
  if(length(missing.inds)>0){
    mat = matrix(apply(df,1,mean.na),nrow(df),ncol(df),byrow=F)
    df[missing.inds] = mat[missing.inds]
  }
  df_rank <- apply(df,2,rank,ties.method="min")
  df_sorted <- data.frame(apply(df, 2, sort))
  df_mean <- apply(df_sorted, 1, mean)
  
  index_to_mean <- function(my_index, my_mean){
    return(my_mean[my_index])
  }
  
  df_final <- apply(df_rank, 2, index_to_mean, my_mean=df_mean)
  rownames(df_final) <- rownames(df)
  
  if(length(missing.inds)>0)
    df_final[missing.inds] = NA
  
  return(df_final)
}

rowConcat <- function(lm)
{
  M <- length(lm)
  mat <- lm[[1]]
  if(M > 1)
    for(m in 2:M)
      mat <- cbind(mat,lm[[m]])
  return(mat)
}