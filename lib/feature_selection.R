################## Part2: SIFT feature selection
# three methods:
# 1. based on variance(cut off)
# 2. PCA
# 3. Random forest to choose important features

#############################     method1: based on variance     #############################
data <- sift
cutoff <- 0.5e-6

variance_cut_off <- function(data, cutoff){
  # cutoff: the variance cutoff value
  
  variance <-  apply(data, 2, var)
  min(variance)
  max(variance)
  
  # count the number of variables left
  # if variance > 0.5e-6, we keep the feature, then there are 1968(n) features remaining
  n=0
  for(i in 1:5000){
    if(variance[i] >= cutoff){n=n+1}
    else{n=n}
  }
  
  # remove the features with small variance
  cut <- rep(cutoff,5000)
  getcol <-  variance - cut >= 0
  return(data[,getcol])
}

#############################     method2: PCA     #############################

# princomp() can only be used with more units than variables
# use prcomp()
# !! We can only get 2000 principle components

pca <- function(data, n){
  # n: the number of principle components we want to keep
  
  pca=prcomp(data, center=TRUE, scale=TRUE);
  
  # Cum.Screeplot.
  pr_var=(pca$sdev)^2;
  plot(cumsum(pr_var)/sum(pr_var)*100,ylim=c(0,100),type="b",xlab="component",ylab="c umulative propotion (%)",main="Cum. Scree plot");
  plot(pca)
  
  return(pca$x[,1:n])
}

