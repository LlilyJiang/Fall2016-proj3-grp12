########################
### Cross Validation ###
########################

### Project 3 Group 12

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

############################## cv.function for xgboost ##########################################

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

################# cv.function for SGD logistic regression ##########################################

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


################# cv.function for new xgboost: use the function from xgboost package  ##########################################


# Simple cross validated xgboost training function (returning minimum error for grid search)
# 5 fold

xgbCV <- function (params) {
  fit <- xgb.cv(
    data = data.matrix(dat_train), 
    label = label_train, 
    param =params, 
    missing = NA, 
    nfold = folds, 
    prediction = FALSE,
    early.stop.round = 50,
    maximize = FALSE,
    nrounds = nrounds
  )
  rounds <- nrow(fit)
  metric = paste('test.',eval,'.mean',sep='')
  idx <- which.min(fit[,fit[[metric]]]) 
  val <- fit[idx,][[metric]]
  res <<- rbind(res,c(idx,val,rounds))
  colnames(res) <<- c('idx','val','rounds')
  return(val)
}










