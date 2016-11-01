# test functions of models we tried but not used in final
# including:
# Gbm(bourulli), svm, gbm(adaboost)


################################ test() for GBM (distribution = bernoulli) ########################################

test <- function(fit_train, dat_test){
  
  ### Fit the classfication model with testing data
  
  ### Input: 
  ###  - the fitted classification model using training data
  ###  -  processed features from testing images 
  ### Output: training model specification
  
  ### load libraries
  library("gbm")
  
  pred <- predict(fit_train$fit, newdata=dat_test, 
                  n.trees=fit_train$iter, type="response")
  
  return(as.numeric(pred> 0.5))
}

################################ test() for Adaboosting (distribution = adaboost) ########################################

test.ada <- function(fit_train, dat_test){
  
  ### Fit the classfication model with testing data
  
  ### Input: 
  ###  - the fitted classification model using training data
  ###  -  processed features from testing images 
  ### Output: training model specification
  
  ### load libraries
  library("gbm")
  
  pred <- predict(fit_train$fit, newdata=dat_test, 
                  n.trees=fit_train$iter, type="response")
  
  return(as.numeric(pred> 0.5))
}

################################ testSVM() for SVM ########################################

testSVM <- function(fit_train, dat_test){
  
  ### Fit the classfication model with testing data
  
  ### Input: 
  ###  - the fitted classification model using training data
  ###  -  processed features from testing images 
  ### Output: training model specification
  
  ### load libraries
  library(e1071)
  
  pred <- predict(fit_train, newdata=dat_test)
  
  return(as.numeric(pred> 0.5))
}
