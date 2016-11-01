#########################################################
### Train a classification model with training images ###
#########################################################

### Project 3 Group 12

train <- function(dat_train, label_train, par=NULL){
  
  ### Train a Gradient Boosting Model (GBM) using processed features from training images
  
  ### Input: 
  ###  -  processed features from images 
  ###  -  class labels for training images
  ### Output: training model specification
  
  ### load libraries
  library("gbm")
  
  ### Train with gradient boosting model
  if(is.null(par)){
    ### default parameter values
    depth <- 6
    Ntrees <- 1000
    Shrinkage <- 0.01
  } else {
    depth <- par$depth
    Ntrees <- par$Ntrees
    Shrinkage <- par$Shrinkage
  }
  
  fit_gbm <- gbm.fit(x=dat_train, y=label_train,
                     n.trees=Ntrees,
                     distribution="bernoulli",
                     interaction.depth=depth,
                     shrinkage=Shrinkage,
                     bag.fraction = 0.5,
                     verbose=FALSE)
  best_iter <- gbm.perf(fit_gbm, method="OOB")
  # best_iter <- gbm.perf(fit_gbm, 
  #               plot.it = TRUE, 
  #               oobag.curve = TRUE, 
  #               overlay = TRUE, 
  #               method = c("OOB","test")[1])
  

  return(list(fit=fit_gbm, iter=best_iter))
}

################################ train.ada() for Adaboosting ########################################

train.ada <- function(dat_train, label_train, par=NULL){
  
  ### Train a Gradient Boosting Model (GBM) using processed features from training images
  
  ### Input: 
  ###  -  processed features from images 
  ###  -  class labels for training images
  ### Output: training model specification
  
  ### load libraries
  library("gbm")
  
  ### Train with gradient boosting model
  if(is.null(par)){
    ### default parameter values
    depth <- 6
    Ntrees <- 1000
    Shrinkage <- 0.01
  } else {
    depth <- par$depth
    Ntrees <- par$Ntrees
    Shrinkage <- par$Shrinkage
  }
  
  fit_gbm <- gbm.fit(x=dat_train, y=label_train,
                     n.trees=Ntrees,
                     distribution="adaboost",
                     interaction.depth=depth,
                     shrinkage=Shrinkage,
                     bag.fraction = 0.5,
                     verbose=FALSE)
  best_iter <- gbm.perf(fit_gbm, method="OOB")
  # best_iter <- gbm.perf(fit_gbm, 
  #               plot.it = TRUE, 
  #               oobag.curve = TRUE, 
  #               overlay = TRUE, 
  #               method = c("OOB","test")[1])
  
  
  return(list(fit=fit_gbm, iter=best_iter))
}

#distribution: a description of the error distribution to be used in the
#          model. Currently available options are "gaussian" (squared
#          error), "laplace" (absolute loss), "bernoulli" (logistic
#          regression for 0-1 outcomes), "adaboost" (the AdaBoost
#          exponential loss for 0-1 outcomes), "poisson" (count
#          outcomes), and "coxph" (censored observations). 


# plot.it: an indicator of whether or not to plot the performance
#         measures. Setting 'plot.it=TRUE' creates two plots. The first
#         plot plots  'object$train.error' (in black) and
#         'object$valid.error' (in red)  versus the iteration number.
#         The scale of the error measurement, shown on the  left
#         vertical axis, depends on the 'distribution' argument used in
#         the initial call to 'gbm'.

# oobag.curve: indicates whether to plot the out-of-bag performance
#         measures in a second plot.
# 
# overlay: if TRUE and oobag.curve=TRUE then a right y-axis is added to
#         the training and test error plot and the estimated
#         cumulative improvement in the loss  function is plotted
#         versus the iteration number.
# 
# method: indicate the method used to estimate the optimal number of
#         boosting iterations. 'method="OOB"' computes the out-of-bag
#         estimate and 'method="test"' uses the test (or validation)
#         dataset  to compute an out-of-sample estimate. method="cv" 
#         extracts the optimal number of iterations using cross-validation 
#         if gbm was called with cv.folds>1

################################ trainSVM() for SVM ########################################

trainSVM <- function(dat_train, label_train, par = NULL){
  
  ### Train a SVM using processed features from training images
  
  ### Input: 
  ###  -  processed features from images 
  ###  -  class labels for training images
  ### Output: training model specification
  
  ### load libraries
  library(e1071)
  
  ### Train with gradient boosting model
  if(is.null(par)){
    ### default parameter values
    gamma <- 0.1
    cost <- 1
    kernel <- 'radial'
  } else {
    gamma <- par$gamma
    cost <- par$cost
    kernel <- par$kernel
  }
  
  fit_svm <- svm(dat_train, label_train, kernel = kernel,gamma = gamma, cost = cost, cross = 10)
  
  return(fit = fit_svm)
}

################################ xg.train() for xgboost ########################################

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


