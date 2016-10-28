#############################################
### Main execution script for experiments ###
#############################################

### Specify directories
#setwd("./proj3_sample")

#img_train_dir <- "./data/zipcode_train/"
#img_test_dir <- "./data/zipcode_test/"



### Train a classification model with training images
source("./lib/train.R")
source("./lib/test.R")

### Model selection with cross-validation
# Choosing between different values of interaction depth for GBM
source("cross_validation.R")
depth_values <- seq(3, 11, 2)
Ntrees_values <- c(200,500,1000,2000,4000)
Shrinkage_values <- c(0.001,0.01,0.1)


################## use cross validation to tune parameters based on all SIFT features################
### Import training images class labels

label_train <- append(rep.int(1,1000),rep.int(0,1000))
dat_train <- read.csv("sift_features.csv")
dat_train <- t(dat_train)
#tm_feature_train <- system.time(dat_train <- feature(img_train_dir, "img_zip_train"))
#tm_feature_test <- system.time(dat_test <- feature(img_test_dir, "img_zip_test"))
#save(dat_train, file="./output/feature_train.RData")
#save(dat_test, file="./output/feature_test.RData")


#err_cv <- array(dim=c(length(depth_values), 2)) 

result_cv <- array(dim=c(length(depth_values), length(Ntrees_values),length(Shrinkage_values)))
K <- 5  # number of CV folds
for(i in 1:length(depth_values)){
  for(j in 1:length(Ntrees_values)){
    for(k in 1:length(Shrinkage_values)){
      cat("i=", i, "\n")
      cat("j=", j, "\n")
      cat("k=", k, "\n")
      result_cv[i,j,k] <- cv.function(dat_train, label_train, depth_values[i],Ntrees_values[j],Shrinkage_values[k], K)
    }
  }
}
#c=cv.function(dat_train, label_train, 9,2000,0.001,K)
save(result_cv, file="./output/err_cv.RData")

# Choose the best parameter value
index_best=which(result_cv==min(result_cv),arr.ind = TRUE)
depth_best <- depth_values[index_best[1]]
Ntrees_best <- Ntrees_values[index_best[2]]
Shrinkage_best <-Shrinkage_values[index_best[3]]
par_best <- list(depth=depth_best,Ntrees=Ntrees_best,Shrinkage=Shrinkage_best)



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

################## end of using cross validation to tune parameters based on all SIFT features################



################## use cross validation to tune parameters based on selected SIFT features################
source("feature_sift.R")

# dat_train, label_train should be changed for different SIFT feature sets
# example 1
dat_train=dat_train1
label_train=label_train1

result_cv <- array(dim=c(length(depth_values), length(Ntrees_values),length(Shrinkage_values)))
K <- 5  # number of CV folds
for(i in 1:length(depth_values)){
  for(j in 1:length(Ntrees_values)){
    for(k in 1:length(Shrinkage_values)){
      cat("i=", i, "\n")
      cat("j=", j, "\n")
      cat("k=", k, "\n")
      result_cv[i,j,k] <- cv.function(dat_train, label_train, depth_values[i],Ntrees_values[j],Shrinkage_values[k], K)
    }
  }
}
#c=cv.function(dat_train, label_train, 9,2000,0.001,K)
save(result_cv, file="/Users/Amyummy/Documents/Rstudio/ads_pro3/err_cv.RData")

# Choose the best parameter value
index_best=which(result_cv==min(result_cv),arr.ind = TRUE)
depth_best <- depth_values[index_best[1]]
Ntrees_best <- Ntrees_values[index_best[2]]
Shrinkage_best <-Shrinkage_values[index_best[3]]
par_best <- list(depth=depth_best,Ntrees=Ntrees_best,Shrinkage=Shrinkage_best)



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






