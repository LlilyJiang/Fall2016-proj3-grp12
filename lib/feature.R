################## SIFT feature selection

# PCA
# Having tried several times of different feature sets, we decide to use 750 features selected by PCA
# So we can only run PCA

# princomp() can only be used with more units than variables
# use prcomp()


pca <- function(data, n){
  # n: the number of principle components we want to keep
  
  pca=prcomp(data, center=TRUE, scale=TRUE);
  
  # Cum.Screeplot.
  pr_var=(pca$sdev)^2;
  plot(cumsum(pr_var)/sum(pr_var)*100,ylim=c(0,100),type="b",xlab="component",ylab="c umulative propotion (%)",main="Cum. Scree plot");
  plot(pca)
  abline(h=80,col="red")
  # dim(pca$x[,1:n2])
  
  return(pca$x[,1:n])
}


