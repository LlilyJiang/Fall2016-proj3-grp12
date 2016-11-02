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
source("Fall2016-proj3-grp12/lib/feature_sift.r")
data_train <- pca(sift, 750)
source("Fall2016-proj3-grp12/lib/Other methods we tried/feature.WeTried.R")
data_train <- variance_cut_off (sift, 0.5e-6)
data_train <- random_forest(sift, label_train, 750)


### MODEL CONSTRUCTION ###

# Train a classification model with training images
source("Fall2016-proj3-grp12/lib/train.r")
source("Fall2016-proj3-grp12/lib/test.r")

# Model selection with cross-validation
source("Fall2016-proj3-grp12/lib/Other methods we tried/cross_validation.WeTried.R")


# Choose the best parameter value
index_best <- which(result_cv == min(result_cv), arr.ind = TRUE)

