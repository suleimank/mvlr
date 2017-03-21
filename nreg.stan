# Bayesian Multi-view Multi-task Linear Regression
# Copyright (R), all rights reserved by authors.
#
# Authors: 
# Suleiman Khan (suleiman.khan@helsinki.fi)
# Muhammad Ammad (muhammad.ammad-ud-din@helsinki.fi )

data {
  int nY;
  int dX;
  int<upper=dX> dY;
  matrix[nY,dY] Y;
  matrix[nY,dX] X;
  int M;
  int<lower=0> dXm[M];
  real<lower=0> tau_v;
  real<lower=0> lambda_v;
  vector[dY] wp;
  vector[M] taup;
}
transformed data {
 int dXmINDS[M,2];

 dXmINDS[1,1] <- 1;
 for(m in 1:M){
  if(m>1){
    dXmINDS[m,1] <- dXmINDS[m-1,2]+1;
  }
  dXmINDS[m,2] <- dXmINDS[m,1]+dXm[m]-1;  
 }
}
parameters {
  vector<lower=-3,upper=3>[dX] beta; 
  vector<lower=0,upper=M>[dX] lambda; 
  simplex[M] tau;
  simplex[dY] W;
  vector<lower=0,upper=1>[dY] sigma;
}
transformed parameters {
  matrix[nY,dY] mu; 
  mu <- X*beta*W';
}
model{
  tau ~ dirichlet(taup); 
  lambda ~ cauchy(0,lambda_v); 
  for(iM in 1:M){
    beta[dXmINDS[iM,1]:dXmINDS[iM,2]] ~ cauchy(0, lambda[dXmINDS[iM,1]:dXmINDS[iM,2]] * tau[iM]);
  }
  W ~ dirichlet(wp); 
  sigma ~ cauchy(0,1); 
  for(idY in 1:dY)
  {
    Y[,idY] ~ normal(mu[,idY],sigma[idY]);
  }
}