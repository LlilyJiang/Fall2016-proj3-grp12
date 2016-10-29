########################
### Cross Validation ###
########################

### Project 3 Group 12

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
    # print('trained')
    pred <- test(fit, test.data)  
    # print('tested')
    cv.error[i] <- mean(pred != test.label)
  }
  
  return(mean(cv.error))
}

cvsvm.function <- function(data, label, k,c,g, K){
  # data: the whole dataset
  # label: a column vector with 0 and 1
  # K: number of folds during the cross validation process
  # c: cost_values
  # g: gamma_values
  # k: kernel_values
  
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
    
    par <- list(kernel = k,cost = c, gamma = g)
    fit <- trainSVM(train.data, train.label, par)
    # print('trained')
    pred <- testSVM(fit, test.data)  
    # print('tested')
    cv.error[i] <- mean(pred != test.label)
  }
  
  return(mean(cv.error))
}


