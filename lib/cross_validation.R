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

  install.packages("caret")
  library(caret)
  flds <- createFolds(1:dim(data)[1], K, list=T, returnTrain=F)
  
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
