######################################################
### Fit the classification model with testing data ###
######################################################

### Project 3 Group 12

########################################## test functions for baseline model ############################################

xg.test.new <- function(fit_train.new, dat_test){
  
  ### Fit the classfication model with testing data
  
  ### Input: 
  ###  - the fitted classification model using training data
  ###  -  processed features from testing images 
  ### Output: prediction
  
  ### load libraries
  library("xgboost")
  
  # fit_train.new=xg.train.new()
  pred <- predict(fit_train.new$fit, as.matrix(dat_test))
  
  return(as.numeric(pred> 0.5))
}

########################################## test functions for advanced model ############################################
sgd.test <- function(fit_train.new, dat_test){
  
  ### Fit the classfication model with testing data
  
  ### Input: 
  ###  - the fitted classification model using training data
  ###  -  processed features from testing images 
  ### Output: prediction
  
  ### load libraries
  library("sgd")
  
  pred <- predict(fit_train.new, dat_test,type = 'response')  
  pred <- ifelse(pred <= 0.5, 0, 1) 
 
  
  return(pred)
}
