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

cvsvm.function <- function(data, label, k, c, g, K){
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

cvxgboost.function <- function(data, label, d, n, r, K){
  # data: the whole dataset
  # label: a column vector with 0 and 1
  # K: number of folds during the cross validation process
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
    fit <- trainxgboost(train.data, train.label, par)
    # print('trained')
    pred <- testxgboost(fit, test.data)  
    # print('tested')
    cv.error[i] <- mean(pred != test.label)
  }
  
  return(mean(cv.error))
}

############ cv.function for xgboost ##########################################

xg.cv.function <- function(X.train, y.train, d,n,r,p, K){
  
  n <- length(y.train)
  n.fold <- floor(n/K)
  s <- sample(rep(1:K, c(rep(n.fold, K-1), n-(K-1)*n.fold)))  
  cv.error <- rep(NA, K)
  
  for (i in 1:K){
    print(i)
    train.data <- X.train[s != i,]
    train.label <- y.train[s != i]
    test.data <- X.train[s == i,]
    test.label <- y.train[s == i]
    
    par <- list(max_depth=d,nround=n,eta=r,colsample=p)
    fit <- xg.train(train.data, train.label, par)
    print('trained')
    pred <- xg.test(fit, test.data)  
    print('tested')
    cv.error[i] <- mean(pred != test.label)  
    
  }			
  #return(c(mean(cv.error),sd(cv.error)))
  return(mean(cv.error))
}

############ cv.function for SGD logistic regression ##########################################
cvsgd.function <- function(data, label,K){
  # data: the whole dataset
  # label: a column vector with 0 and 1
  # K: number of folds during the cross validation process

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
    
    
    #fit <- sgd(train.data, train.label,model='glm',model.control=list(family="binomial",lambda2=0.001))
    fit <- sgd(train.data, train.label,model='glm',model.control=binomial(link="logit"))
    # print('trained')
    pred <- predict(fit, test.data,type = 'response')  
    pred <- ifelse(pred <= 0.5, 0, 1) 
    cv.error[i] <- mean(pred != test.label)
  }
  
  return(mean(cv.error))
}



