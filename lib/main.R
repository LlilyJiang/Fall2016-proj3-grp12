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
depth_values <- seq(3, 11, 2)
Ntrees_values <- c(200, 500, 1000, 2000, 4000)
Shrinkage_values <- c(0.01, 0.05, 0.1)

result_cv <- array(dim=c(length(depth_values), length(Ntrees_values),length(Shrinkage_values)))

# Set the number of folds for the cross validation
K <- 5

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






