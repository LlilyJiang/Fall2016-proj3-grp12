######################################################
### Fit the classification model with testing data ###
######################################################

### Project 3 Group 12

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


testxgboost <- function(fit_train, dat_test){
  
  pred <- predict(fit_train, dat_test)
  
  return(as.numeric(pred > 0.5))
}

