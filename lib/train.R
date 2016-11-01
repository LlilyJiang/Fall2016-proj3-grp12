#########################################################
### Train a classification model with training images ###
#########################################################

### Project 3 Group 12

################################################    Baseline Model   ########################################################

### xg.train() for xgboost ###

xg.train <- function(dat_train, label_train, par=NULL){
  
  ### Train a Gradient Boosting Model (GBM) using processed features from training images
  
  ### Input: 
  ###  -  processed features from images 
  ###  -  class labels for training images
  ### Output: training model specification
  
  ### load libraries
  library("xgboost")
  
  ### Train with gradient boosting model
  if(is.null(par)){
    max_depth <- 6
    nround <- 25
    eta <- 0.3
    colsample_bytree <- 1
  } else {
    max_depth <- par$max_depth
    nround <- par$nround
    eta <- par$eta
    colsample <- par$colsample
  }
  
  fit_xgb <- xgboost(data = dat_train, 
                 label = label_train, 
                 eta = eta,
                 max_depth = max_depth, 
                 nround=nround, 
                 subsample = 0.5,
                 colsample_bytree = colsample,
                 seed = 9,
                 eval_metric = "error",
                 objective = "binary:logistic",
                 verbose=FALSE
  )
  
  return(list(fit=fit_xgb))
}

################################### new train function for xgboost: tune more parameters ################
xg.train.new <- function(dat_train, label_train, par=NULL){
  
  ### load libraries
  library("xgboost")
  
  if(is.null(par)){
    par = par0
  } else {
    par = par
  }
  
  fit_xgb <- xgboost(
    data = xgb.DMatrix(data.matrix(dat_train),missing=NaN, label = label_train),
    param = par,
    nrounds = results[which.min(results[,2]),1],
    verbose=FALSE
  )
  
  return(list(fit=fit_xgb))
}

################################################    Baseline Model   ########################################################
