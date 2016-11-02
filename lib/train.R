#########################################################
### Train a classification model with training images ###
#########################################################

### Project 3 Group 12

sift_features <- read.csv("~/Desktop/sift_features.csv")
data_train <- t(sift_features)
label_train <- matrix(c(rep(1,1000),rep(0,1000)), ncol = 1)

################################################    Baseline Model   ########################################################
nround = 500

xg.train <- function(data_train, label_train, par=NULL){
  
  ### load libraries
  library("xgboost")
  
  ### set the default parameters
  ### the optimal parameters we obtained from training dataset
  par.optimal = list(
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
    par = par.optimal
  } else {
    par = par
  }
  
  fit_xgb <- xgboost(
    data = xgb.DMatrix(data.matrix(data_train), missing= NaN, label = label_train),
    param = par,
    # for here, nrounds is dependent, we should follow the code step by step
    # or you can change the nrounds = 1000
    nrounds = nround,
    verbose = TRUE
  )
  
  return(fit_xgb)
}

fit = xg.train(data_train,label_train)

###### cross validation procedure for the base line model ######

# xgboost task parameters
folds <- 5

# Parameter grid to search
eta = c(0.1,0.01)
max_depth = c(4,6,8)
max_delta_step = c(0,1)
subsample = c(0.5,1)
scale_pos_weight = c(0,1)
min_child_weight = c(1,3,6)
nround = 500

result_cv <- array(dim=c(length(eta),length(max_depth),length(max_delta_step),
                         length(subsample),length(scale_pos_weight),length(min_child_weight)))

for(i in 1:length(eta)){
  for(j in 1:length(max_depth)){
    for(k in 1:length(max_delta_step)){
      for (l in 1:length(subsample)){
        for (m in 1:length(scale_pos_weight)){
          for (n in 1:length(min_child_weight)){
          model <- xgb.cv(data = data_train,label = label_train, eta = eta[i],
                           max_depth = max_depth[j], max_delta_step = max_delta_step[k],
                           subsample = subsample[l], sclae_pos_weight = scale_pos_weight[m],
                           min_child_weight = min_child_weight[n], eval_metric = "error",
                           objective = "binary:logistic", nfold = folds,
                           nround = nround, verbose=TRUE)
            rounds <- nrow(model)
            metric <- paste('test.',eval,'.mean',sep='')
            idx <- which.min(model[,model[[metric]]]) 
            val <- model[idx,][[metric]]
            result_cv[i,j,k,l,m,n] <- val
          }
        }
      }
    }
  }
}

# select the best parameters by comparing testing error among all kinds of combinations
index_best <- which(result_cv == min(result_cv), arr.ind = TRUE)
eta.optimal <- eta[index_best[1]]
max_depth.optimal <- max_depth[index_best[2]]
max_delta_step.optimal <-max_delta_step[index_best[3]]
subsample.optimal <- subsample[index_best[4]]
scale_pos_weight.optimal <-scale_pos_weight[index_best[5]]
min_child_weight .optimal<-min_child_weight[index_best[6]]

save(result_cv, file="./output/err_cv_xgboost.RData")

=======
>>>>>>> origin/master

################################################    Advanced Model   ########################################################

sgd.train <- function(dat_train, label_train){
  
  ### load libraries
  library("sgd")
  sgd_fit<- sgd(dat_train,label_train,model='glm',model.control=binomial(link="logit"))
  
  return(sgd_fit)
}


### cv.sgd.function for SGD logistic regression ###
K <- 5

cv.sgd.function <- function(data, label,K){
  # data: the whole dataset
  # label: a column vector with 0 and 1
  # K: number of folds during the cross validation process
  
  set.seed(0)
  library(caret)
  install.packages("sgd")
  library(sgd)
  fold <- createFolds(1:dim(data)[1], K, list=T, returnTrain=F)
  fold <- as.data.frame(fold)
  
  cv.error <- rep(NA, K)
  
  for (i in 1:K){
    test.data <- data[fold[,i],]
    train.data <- data[-fold[,i],]
    test.label <- label[fold[,i],]
    train.label <- label[-fold[,i],]
    
    
    #fit <- sgd(train.data, train.label,model='glm',model.control=list(family="binomial",lambda2=0.001))
    fit <- sgd(train.data, train.label,model='glm',model.control=binomial(link="logit"))
    # print('trained')
    pred <- predict(fit, test.data,type = 'response')  
    pred <- ifelse(pred <= 0.5, 0, 1) 
    cv.error[i] <- mean(pred != test.label)
  }
  
  return(mean(cv.error))
}





