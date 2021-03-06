#####################################################
### Other classification methods we have tried #####
#####################################################

# including:
# gbm(bornulli), svm, gbm(adboosting), logistic regression 

################################ train.ada() for Adaboosting ########################################

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

################################ train.logit() for Logistic Regression ########################################

train.logit <- function(dat_train, label_train){
  library("sgd")
  
  fit <- sgd(dat_train, label_train, model='glm', model.control=binomial(link="logit"))
  
  return(fit)
}

