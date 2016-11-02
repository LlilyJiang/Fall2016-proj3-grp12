######################################################
### Fit the classification model with testing data ###
######################################################

### Project 3 Group 12

########################################## test functions for baseline model ############################################

# load the test data and transform it using the loading matrix we calculated from training data set
load("/Users/jiwenyou/Desktop/Fall2016-proj3-grp12/lib/sift_pca_loading.rda")
sift.features_test <- read.csv("~/Downloads/Project3_poodleKFC_test/sift features_test.csv")
data_test <- as.matrix(t(sift.features_test))
data_test <- data_test %*% load
dim(data_test)

xg.test <- function(fit_train, data_test){
  
  ### Fit the classfication model with testing data
  
  ### Input: 
  ###  - the fitted classification model using training data
  ###  -  processed features from testing images 
  ### Output: prediction
  
  ### load libraries
  library("xgboost")
  
  # fit_train.new=xg.train.new()
  pred <- predict(fit, as.matrix(data_test))
  
  return(as.numeric(pred> 0.5))
}

test_result <- xg.test(fit, data_test)
write.csv(test_result, file= "/Users/jiwenyou/Desktop/Fall2016-proj3-grp12/output/base_model_predict.csv")

########################################## test functions for advanced model ############################################

sgd.test <- function(fit_train, data_test){
  
  ### Fit the classfication model with testing data
  
  ### Input: 
  ###  - the fitted classification model using training data
  ###  -  processed features from testing images 
  ### Output: prediction
  
  ### load libraries
  library("sgd")
  
  pred <- predict(fit_train, data_test,type = 'response')  
  pred <- ifelse(pred <= 0.5, 0, 1) 
 
  
  return(pred)
}
