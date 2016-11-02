#############################################
### Main execution script for experiments ###
#############################################

### Project 3 Group 12

# Specify directories
setwd("./Fall2016-proj3-grp12/lib") # where you put all the R files

# SIFT FEATURE EXTRACTION: use feature_sift.r

# new xgboost(): whole progress, run it step by step

# source all the functions you need
source("./lib/train.r")
source("./lib/test.r")

source("./lib/feature_sift.R")

##################    Base model: use cross validation to tune parameters based on selected SIFT features   ################

# input original SIFT data
sift.feature=read.csv("./data/sift_features.csv")
sift=t(sift.feature)
# dim(sift)

# add lables: 0 for dog and 1 for fried chicken
# use PCA() in feature_sift.R to reduce dimension
data = pca(sift,750)
label1=append(rep(1,1000),rep(0,1000))
# selected data with labels
data=cbind(data,label1)

# dim(data)
# selected data with labels

# randomly select 1600 images as traing set, remaining is the test set.

n2=750 
sam=sample(1:2000,1600)
train_data=data[sam,]
dat_train=train_data[,1:n2]
label_train=train_data[,n2+1]

test_data=data[-sam,]
dat_test=test_data[,1:n2]
label_test=test_data[,n2+1]

# xgboost task parameters
nrounds <- 1000
folds <- 5
obj <- 'binary:logistic'
eval <- 'logloss'

# Parameter grid to search
params <- list(
  eval_metric = eval,
  objective = obj,
  # eta: Analogous to learning rate in GBM (Typical final values to be used: 0.01-0.2)
  eta = c(0.1,0.01),
  max_depth = c(2,4,6,8,10),
  max_delta_step = c(0,1),
  subsample = c(0.5,0.8,1),
  scale_pos_weight = c(0,1),
  min_child_weight = c(1,3,6,9)
)


# Table to track performance from each worker node
res <- data.frame()

library(xgboost)
# Simple cross validated xgboost training function (returning minimum error for grid search)
# 5 fold
xgbCV <- function (params) {
  library(xgboost)
  fit <- xgb.cv(
    data = data.matrix(dat_train), 
    label = label_train, 
    param =params, 
    missing = NA, 
    nfold = folds, 
    prediction = FALSE,
    early.stop.round = 50,
    maximize = FALSE,
    nrounds = nrounds
  )
  rounds <- nrow(fit)
  metric = paste('test.',eval,'.mean',sep='')
  idx <- which.min(fit[,fit[[metric]]]) 
  val <- fit[idx,][[metric]]
  res <<- rbind(res,c(idx,val,rounds))
  colnames(res) <<- c('idx','val','rounds')
  return(val)
}

# !! if you want to install caret, your R version should be >= 3.2.5
library(caret)
# install.packages("NMOF")
library(gbm)
library(NMOF)

# Find minimal testing error in parallel
cl <- makeCluster(round(detectCores()/2)) 
clusterExport(cl, c("xgb.cv",'dat_train','label_train','nrounds','res','eval','folds'))

# this steps takes about 5 hours
sol <- gridSearch(
  fun = xgbCV,
  levels = params,
  method = 'snow',
  cl = cl,
  keepNames = TRUE,
  asList = TRUE
)

# Combine all model results
# install.packages("plyr")
library(plyr)
comb=clusterEvalQ(cl,res)
results <- ldply(comb,data.frame)
stopCluster(cl)

# get and save the best params set.
params <- c(sol$minlevels,objective = obj, eval_metric = eval)
# print and save the params
print(params)
par0 = params 

# Train model given solution (params) above
xgbModel <- xgboost(
  data = xgb.DMatrix(data.matrix(dat_train),missing=NaN, label = label_train),
  param = params,
  nrounds = results[which.min(results[,2]),1],
  verbose=FALSE
)



# run xg.train.new and xg.test.new. where we have already soured those files
#fit_train.new=xg.train.new(dat_train,label_train,par1)
fit_train.new=xg.train.new(dat_train,label_train,params)
# system.time(fit_train.new <- xg.train.new(dat_train,label_train,params))

# fit_train.new=xg.train.new(dat_train,label_train,params)
pred=xg.test.new(fit_train.new, dat_test)
set.seed(708)
save(pred, file="./output/base.test.pred.RData")



#######  this part is for new SIFT data. use the trained model to do the prediction ### no need to run this part for training
# the new image sets must be 2000!
# newdata = ...
# load the rotation of the training data (name is load)
load("./lib/sift_pca_loading.rda")
# dim newdata should be 2000*5000. 
data.pca.new = as.matrix(newdata)%*%load
pred.new = xg.test.new(fit_train.new, data.pca.new)

######  end of no need to run this part for training



# This part is optional: calculate accuracy

accu <- function(label_test,pred_test){
  x=label_test-pred
  n=length(pred)
  k=0
  for(i in 1:n){
    if(x[i]==0){
      k=k+1
    }else{k=k}
  }
  accuracy=k/n
  return(accuracy)
}

accu(label_test,pred)


###################################  Advanced Model :  based on selected CAFFE features ####################################

# data from the result of feature.py with features extracted by Caffe and selected by PCA

# read the selected layer data
caffe = read.csv("./data/fs8.new.csv")
caffe = caffe[,-1]
label2=append(rep(1,1000),rep(0,1000))
data2=cbind(caffe,label2)
dim(data2)

sam=sample(1:2000,1600)

train_data2=data2[sam,]
dat_train=train_data2[,1:1000]
label_train=train_data2[,1001]

test_data2=data2[-sam,]
dat_test=test_data2[,1:1000]
label_test=test_data2[,1001]



##################    Advance model:  on selected Caffe features   ################
# data from the result of feature.py with features extracted by Caffe and selected by PCA


## SGD on logistic regression
library(sgd)

label_train <- matrix(c(rep(1,1000),rep(0,1000)), ncol = 1)
# read the selected layer data
data_train <- read.csv("./data/data_fc8_pca.csv")
data_train <-as.matrix(data_train[,-1])

t<-sample(1:2000,1600)

train.data <- data_train[t,]
test.data <- data_train[-t,]
train.label <- matrix(label_train[t,])
test.label <- matrix(label_train[-t,])


sgd_fit<- sgd(train.data, train.label,model='glm',model.control=binomial(link="logit"))
pred <- predict(sgd_fit, test.data,type = 'response')  
pred <- ifelse(pred <= 0.5, 0, 1) 

#cv.error <- mean(pred != test.label)
#print(cv.error)

save(pred, file="./output/advance.test.pred.RData")


