#########################################################
### Train a classification model with training images ###
#########################################################

### Project 3 Group 12

################################################    Baseline Model   ########################################################

### new train function for xgboost: tune more parameters ###

xg.train.new <- function(dat_train, label_train, par=NULL){
  
  ### load libraries
  library("xgboost")
  
  par1 = list(
  eval_metric = "logloss",
  objective = "binary:logistic",
  # eta: Analogous to learning rate in GBM (Typical final values to be used: 0.01-0.2)
  eta = 0.1,
  max_depth = 6,
  max_delta_step = 1,
  subsample = 0.8,
  scale_pos_weight = 1,
  min_child_weight = 3
)
  
  if(is.null(par)){
    par = par1
  } else {
    par = par
  }
  
  fit_xgb <- xgboost(
    data = xgb.DMatrix(data.matrix(dat_train),missing=NaN, label = label_train),
    param = par,
    # for here, nrounds is dependent, we should follow the code step by step
    # or you can chage the nrounds = 1000
    nrounds = results[which.min(results[,2]),1],
    verbose=FALSE
  )
  
  return(list(fit=fit_xgb))
}


################################################    Advanced Model   ########################################################

sgd.train <- function(dat_train, label_train){
  
  ### load libraries
  library("sgd")
  sgd_fit<- sgd(dat_train,label_train,model='glm',model.control=binomial(link="logit"))
  
  
  
  return(sgd_fit)
}



