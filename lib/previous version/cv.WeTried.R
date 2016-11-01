# cross validation functions of models we tried but not used in the final



################################ cv.function for GBM ########################################

cv.function <- function(data, label, d, n, r, K){
  # data: the whole dataset
  # label: a column vector with 0 and 1
  # K: number of folds during the cross validation process
  # d: depth_values
  # n: Ntrees_values
  # r: Shrinkage_values
  
  set.seed(0)
  library(caret)
  fold <- createFolds(1:dim(data)[1], K, list=T, returnTrain=F)
  fold <- as.data.frame(fold)
  
  cv.error <- rep(NA, K)
  
  for (i in 1:K){
    test.data <- data[fold[,i],]
    train.data <- data[-fold[,i],]
    test.label <- label[fold[,i],]
    train.label <- label[-fold[,i],]
    
    par <- list(depth = d, Ntrees = n, Shrinkage = r)
    fit <- train(train.data, train.label, par)
    print("fit")
    pred <- test(fit, test.data)  
    print("test")
    cv.error[i] <- mean(pred != test.label)
  }
  
  return(mean(cv.error))
}

################################ cv.function for Adaboost ########################################

cv.function <- function(data, label, d, n, r, K){
  # data: the whole dataset
  # label: a column vector with 0 and 1
  # K: number of folds during the cross validation process
  # d: depth_values
  # n: Ntrees_values
  # r: Shrinkage_values
  
  set.seed(0)
  library(caret)
  fold <- createFolds(1:dim(data)[1], K, list=T, returnTrain=F)
  fold <- as.data.frame(fold)
  
  cv.error <- rep(NA, K)
  
  for (i in 1:K){
    test.data <- data[fold[,i],]
    train.data <- data[-fold[,i],]
    test.label <- label[fold[,i],]
    train.label <- label[-fold[,i],]
    
    par <- list(depth = d, Ntrees = n, Shrinkage = r)
    fit <- train.ada(train.data, train.label, par)
    pred <- test.ada(fit, test.data)  
    cv.error[i] <- mean(pred != test.label)
  }
  
  return(mean(cv.error))
}
