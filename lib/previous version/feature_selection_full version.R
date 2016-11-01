################## SIFT feature selection
# three methods:
# 1. based on variance(cut off)
# 2. PCA
# 3. Random forest to choose important features


#############################     method1: based on variance     #############################

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

# Having tried several times of different feature sets, we decide to use 750 features selected by PCA
# So we can only run PCA


#############################     method2: PCA     #############################

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
  
  data.m2=cbind(pca$x[,1:n2],label1)
  # dim(data.m2)
  save(data.m2, file="./lib/data.m2.RData")
  
  # new feature set selected by method1
  sam2=sample(1:dim(data)[1],0.7*dim(data)[1])
  train_data=data.m2[sam2,]
  dat_train=train_data[,1:n]
  label_train=train_data[,n+1]
  
  test_data=data.m2[-sam2,]
  dat_test=test_data[,1:n]
  label_test=test_data2[,n+1]
  save(dat_train, file="./lib/dat_train.RData")
  save(label_train, file="./lib/label_train.RData")
  save(dat_test, file="./lib/dat_test.RData")
  save(label_test, file="./lib/label_test.RData")
  
  return(pca$x[,1:n])
}


#############################     method3: random forest     ################################
# since the permutation variable importance is affected by collinearity
# it's necessary to handle collinearity prior to running random forest for extracting important variables.

########### 1.deal with collinearity
# sessionInfo(): R version 3.2.2 (caret only available for version > 3.2.5)

# load required libraries

random_forest <- function(data, label, n){
  # n: number of features to keep
  
  install.packages("caret", dependencies = c("Depends", "Suggests"))
  library(caret)
  install.packages("corrplot")
  library(corrplot)
  library(plyr)
  
  # Give each feature a "name" and Calculate correlation matrix
  feature1 <- data
  colnames(feature1)[1:5000] <- as.character(seq(1,5000,by=1))
  descrCor <- cor(feature1)
  
  # Print correlation matrix and look at max correlation
  summary(descrCor[upper.tri(descrCor)])
  
  # Find attributes that are highly corrected
  highlyCorrelated <- findCorrelation(descrCor, cutoff=0.6)
  
  # Print indexes of highly correlated attributes
  print(highlyCorrelated)
  
  # Indentifying Variable Names of Highly Correlated Variables
  highlyCorCol <- colnames(feature1)[highlyCorrelated]
  
  # Print highly correlated attributes
  highlyCorCol
  
  # Remove highly correlated variables and create a new dataset
  features1 <- feature1[, -which(colnames(feature1) %in% highlyCorCol)]
  dim(features1)
  # after remove highly corelated variables, there are still 4913 features remaining.
  
  ########### 2.Use random forest
  # ensure the results are repeatable
  install.packages("randomForest")
  library(randomForest)
  
  df <- as.data.frame(cbind(features1,label))
  allX <- paste("X",1:ncol(features1),sep="")
  names(df) <- c(allX,"label")
  
  #Train Random Forest
  time <- system.time(rf <- randomForest(as.factor(label)~.,data = df, importance = TRUE,ntree = 500))
  
  #Evaluate variable importance
  imp <- importance(rf, type=1)
  imp <- data.frame(predictors=rownames(imp),imp)
  
  # Order the predictor levels by importance
  imp.sort <- arrange(imp,desc(MeanDecreaseAccuracy))
  imp.sort$predictors <- factor(imp.sort$predictors,levels=imp.sort$predictors)
  
  # Select the top n predictors
  imp.100=imp.sort[1:n,]
  print(imp.100)
  
  # Plot Important Variables
  varImpPlot(rf, type=1)
  
  return(df[,c(imp.100$predictors)])
}

