#############################################
### Main execution script for experiments ###
#############################################

### Project 3 Group 12

# Specify directories
setwd("/Users/jiwenyou/Desktop")

### FEATURE EXTRACTION ###

# Import training images and construct visual feature

# Import training feature documents
sift.feature <- read.csv("sift_features.csv")
sift <- t(sift.feature)

# Import the class label
label_train <- matrix(c(rep(1,1000),rep(0,1000)), ncol = 1)

### FEATURE SELECTION ###
source("Fall2016-proj3-grp12/lib/feature_selection.r")
data_train <- variance_cut_off (sift, 0.5e-6)
data_train <- pca(sift, 1000)
data_train <- random_forest(sift, label_train, 100)

data_train <- sift
### MODEL CONSTRUCTION ###

# Train a classification model with training images
source("Fall2016-proj3-grp12/lib/train.r")
source("Fall2016-proj3-grp12/lib/test.r")

# Model selection with cross-validation
source("Fall2016-proj3-grp12/lib/cross_validation.r")

# Set the range for the tunning parameters
depth_values <- seq(3, 10, 1)
Ntrees_values <- c(200, 500, 1000, 1500, 2000, 4000)
Shrinkage_values <- c(0.01, 0.05, 0.1)

result_cv <- array(dim=c(length(depth_values), length(Ntrees_values),length(Shrinkage_values)))

# Set the number of folds for the cross validation
K <- 5

i = 5
j = 5
k = 3

train_time1 <- system.time(train1 <- cv.function(data_train, label_train, depth_values[i], Ntrees_values[j], Shrinkage_values[k], 2))
cat("Time for constructing training features 1 =", train_time1[1], "s \n")
# train1 = 0.3105
# time = 220.216s


train_time2 <- system.time(train2 <- cvxgboost.function(data_train, label_train, depth_values[i],Ntrees_values[j], Shrinkage_values[k], 4))
cat("Time for constructing training features 2 =", train_time2[1], "s \n")
train2
# train2 = 0.4055
# time = 0.897s

for(i in 1:length(depth_values)){
  for(j in 1:length(Ntrees_values)){
    for(k in 1:length(Shrinkage_values)){
      result_cv[i,j,k] <- cv.function(data_train, label_train, depth_values[i], Ntrees_values[j], Shrinkage_values[k], K)
    }
  }
}

save(result_cv, file="./output/err_cv.RData")

# Choose the best parameter value
index_best <- which(result_cv == min(result_cv), arr.ind = TRUE)
depth_best <- depth_values[index_best[1]]
Ntrees_best <- Ntrees_values[index_best[2]]
Shrinkage_best <-Shrinkage_values[index_best[3]]
par_best <- list(depth = depth_best, Ntrees = Ntrees_best, Shrinkage = Shrinkage_best)



### svm ###
kernel_values <- c('radial')
#cost_values <- c(0.01, 0.1, 1, 2.7, 10, 100, 150, 200, 250, 300, 350)
#gamma_values <- c(0.0001, 0.0005, 0.0007, 0,001, 0.01, 0.09, 0.015, 0.02, 0.025, 0.03, 0.1, 1)
cost_values <- c( 50,100,200,500)
gamma_values <- c( 0.001)

result_cv <- array(dim=c(length(kernel_values),length(cost_values),length(gamma_values)))
K <- 5  # number of CV folds
for(i in 1:length(kernel_values)){
  for(j in 1:length(cost_values)){
    for(k in 1:length(gamma_values)){
      cat("i=", i, "\n")
      cat("j=", j, "\n")
      cat("k=", k, "\n")
      result_cv[i,j,k] <- cvsvm.function(data_train[,-ncol(data_train)], label_train, kernel_values[i],cost_values[j],gamma_values[k], K)
    }
  }
}
c=cvsvm.function(data_train[,-ncol(data_train)], label_train, 'radial',0.1,0.01,K)



### logistic regression###
# conduct CV within the function.Output a plot of error rate versus lambda
# didn't call function in other files
library(glmnet)
log.fit<-cv.glmnet(data_train[,-ncol(data_train)], label_train,family = "binomial", type.measure = "class",nfolds=5)
plot(log.fit)

### SGD on logistic regression
library(sgd)

# not tune yet... better than glm
sgd_cvresult<-cvsgd.function(data_train[,-ncol(data_train)], label_train,5)






############################## not changed yet ###################################
# Visualize CV results
pdf("./fig/cv_results.pdf", width=7, height=5)
plot(depth_values, err_cv[,1], xlab="Interaction Depth", ylab="CV Error",
     main="Cross Validation Error", ylim=c(0, 0.45))
points(depth_values, err_cv[,1], col="blue", pch=16)
lines(depth_values, err_cv[,1], col="blue")
arrows(depth_values, err_cv[,1]-err_cv[,2],depth_values, err_cv[,1]+err_cv[,2], 
       length=0.1, angle=90, code=3)
dev.off()


# train the model with the entire training set
tm_train <- system.time(fit_train <- train(dat_train, label_train, par_best))
save(fit_train, file="./output/fit_train.RData")

### Make prediction 
tm_test <- system.time(pred_test <- test(fit_train, dat_test))
save(pred_test, file="./output/pred_test.RData")

### Summarize Running Time
cat("Time for constructing training features=", tm_feature_train[1], "s \n")
cat("Time for constructing testing features=", tm_feature_test[1], "s \n")
cat("Time for training model=", tm_train[1], "s \n")
cat("Time for making prediction=", tm_test[1], "s \n")

##################################### another function xgboost() ###########################################

# parameters to be tuned
max_depth_values <- seq(3, 9, 3)
nround_values <- seq(15, 25, 5)
eta_values <- c(0.0005,0.001,0.01,0.1,0.3)
colsample_values <- c(0.5,1)

dat_train=dat_train2
label_train=label_train2

xg_result_cv <- array(dim=c(length(max_depth_values), length(nround_values),length(eta_values),length(colsample_values)))
K <- 5  # number of CV folds
for(i in 1:length(max_depth_values)){
  for(j in 1:length(nround_values)){
    for(k in 1:length(eta_values)){
      for(p in 1:length(colsample_values)){
      cat("i=", i, "\n")
      cat("j=", j, "\n")
      cat("k=", k, "\n")
      cat("p=", p, "\n")
      xg_result_cv[i,j,k,p] <- xg.cv.function(dat_train, label_train, max_depth_values[i],nround_values[j],eta_values[k],colsample_values[p],K)
       }
    }
  }
}

save(xg_result_cv, file="/Users/Amyummy/Documents/Rstudio/ads_pro3/xg_err_cvT.RData")

# Choose the best parameter value
xg_index_best=which(xg_result_cv==min(xg_result_cv),arr.ind = TRUE)
max_depth_best <- max_depth_values[xg_index_best[1]]
nround_best <- nround_values[xg_index_best[2]]
eta_best <-eta_values[xg_index_best[3]]
colsample_best <-colsample_values[xg_index_best[4]]
xg_par_best <- list(max_depth=max_depth_best,nround=nround_best,eta=eta_best,colsample=colsample_best)








