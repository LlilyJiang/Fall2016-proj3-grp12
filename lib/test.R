######################################################
### Fit the classification model with testing data ###
######################################################

### Project 3 Group 12

########################################## test functions for baseline model ############################################


### xg.test() for xgboost ###
xg.test <- function(fit_train, dat_test){
  
  ### Fit the classfication model with testing data
  
  ### Input: 
  ###  - the fitted classification model using training data
  ###  -  processed features from testing images 
  ### Output: training model specification
  
  ### load libraries
  library("xgboost")
  
  # fit_train=xg.train()
  pred <- predict(fit_train$fit, dat_test)
  
  return(as.numeric(pred> 0.5))
}

### xg.test.new() for xgboost ###
xg.test.new <- function(fit_train.new, dat_test){
  
  library("xgboost")
  pred <- predict(fit_train.new$fit, dat_test)
  
  return(as.numeric(pred> 0.5))
}

########################################## test functions for baseline model ############################################
