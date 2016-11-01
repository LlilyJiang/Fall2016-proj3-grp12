# use data from PCA selection
dat_test=dat_test2
dat_train=dat_train2
label_train=label_train2
label_test=label_test2


# xgboost task parameters
nrounds <- 1000
folds <- 5
obj <- 'binary:logistic'
eval <- 'error'

# Parameter grid to search
params <- list(
  eval_metric = eval,
  objective = obj,
  # eta: Analogous to learning rate in GBM (Typical final values to be used: 0.01-0.2)
  eta = c(0.1,0.01),
  max_depth = c(4,6,8),
  max_delta_step = c(0,1),
  subsample = c(0.5,1),
  scale_pos_weight = c(0,1),
  min_child_weight = c(1,3,6)
)

# Table to track performance from each worker node
res <- data.frame()

# Simple cross validated xgboost training function (returning minimum error for grid search)
# 5 fold
xgbCV <- function (params) {
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

library(caret)
# install.packages("NMOF")
library(gbm)
library(NMOF)
# Find minimal testing error in parallel
cl <- makeCluster(round(detectCores()/2)) 

clusterExport(cl, c("xgb.cv",'dat_train','label_train','nrounds','res','eval','folds'))
sol <- gridSearch(
  fun = xgbCV,
  levels = params,
  method = 'snow',
  cl = cl,
  keepNames = TRUE,
  asList = TRUE
)

# Combine all model results
comb=clusterEvalQ(cl,res)
results <- ldply(comb,data.frame)
stopCluster(cl)

# Train model given solution above
params <- c(sol$minlevels,objective = obj, eval_metric = eval)
xgbModel <- xgboost(
  data = xgb.DMatrix(data.matrix(dat_train),missing=NaN, label = label_train),
  param = params,
  nrounds = results[which.min(results[,2]),1]
)

print(params)
print(results)

#####################
# here we use the result of params as the parameters in xgboost()
xgbresult <- xgboost(
  data = xgb.DMatrix(data.matrix(dat_train),missing=NaN, label = label_train),
  param = params,
  nrounds = results[which.min(results[,2]),1],
  verbose=FALSE
)


par0 = list(
  eval_metric = "error",
  objective = "binary:logistic",
  # eta: Analogous to learning rate in GBM (Typical final values to be used: 0.01-0.2)
  eta = 0.01,
  max_depth = 4,
  max_delta_step = 1,
  subsample = 1,
  scale_pos_weight = 1,
  min_child_weight = 1
)

xg.train.new <- function(dat_train, label_train, par=NULL){
  
  ### load libraries
  library("xgboost")
  
  ### Train with gradient boosting model
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

# fit_train.new=xg.train.new(dat_train2,label_train2,par0)

system.time(fit_train.new <- xg.train.new(dat_train2,label_train2,par0))

xg.test.new <- function(fit_train.new, dat_test){
  library("xgboost")
  pred <- predict(fit_train.new$fit, dat_test)
  return(as.numeric(pred> 0.5))
}

##  calculate accuracy. 

set.seed(100)
fit_train.new=xg.train.new(dat_train,label_train,params)
pred=xg.test.new(fit_train.new, dat_test)

x=label_test-pred
k=0
for(i in 1:600){
  if(x[i]==0){
    k=k+1
  }else{k=k}
}
accu=k/600
accu

