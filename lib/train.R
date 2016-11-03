#########################################################
### Train a classification model with training images ###
#########################################################

### Project 3 Group 12 

################################################    Baseline Model   ########################################################
# Baseline Model
xg.train <- function(data_train, label_train, par=NULL){
  
  ### load libraries
  library("xgboost")
  
  ### set the default parameters
  ### the optimal parameters we obtained from training dataset
  par.optimal = list(
  eval_metric = "logloss",
  objective = "binary:logistic",
  # eta: Analogous to learning rate in GBM (Typical final values to be used: 0.01-0.2)
  eta = 0.01,
  max_depth = 8,
  max_delta_step = 1,
  subsample = 0.8,
  scale_pos_weight = 1,
  min_child_weight = 6
  )
  
  if(is.null(par)){
    par = par.optimal
  } else {
    par = par
  }
  
  fit_xgb <- xgboost(
    data = xgb.DMatrix(data.matrix(data_train), missing= NaN, label = label_train),
    param = par,
    nrounds = 1000,
    verbose = TRUE
  )
  
  return(fit_xgb)
}




################################################    Advanced Model   ########################################################

sgd.train <- function(dat_train, label_train){
  
  ### load libraries
  library("sgd")
  sgd_fit<- sgd(dat_train,label_train,model='glm',model.control=binomial(link="logit"))
  
  return(sgd_fit)
}


### cv.sgd.function for SGD logistic regression, for cross validation ###

cv.sgd.function <- function(data, label,K){
  # data: the whole dataset
  # label: a column vector with 0 and 1
  # K: number of folds during the cross validation process
  
  set.seed(0)
  library(caret)
  #install.packages("sgd")
  library(sgd)
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





