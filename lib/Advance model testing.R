
######## advance model ############

source("./lib/cross_validation.r")

### caffeNorm1
caffeNorm1.feature <- read.csv("data_norm1_pca.csv")
caffeNorm1.feature <-as.matrix(caffeNorm1.feature[,-1])

# SGD logistic regression
sgd_cvresult<-cvsgd.function(caffeNorm1.feature, label_train,5)

log.fit<-cv.glmnet(caffeNorm1.feature, label_train,family = "binomial", type.measure = "class",nfolds=5)
plot(log.fit)

# SVM
c=cvsvm.function(caffeNorm1.feature, label_train, 'radial',100,0.0001,5)

# GBM
c2 <- cv.function(caffeNorm1.feature, label_train, 9, 2000, 0.01, 5)


### caffeNorm2

caffeNorm2.feature <- read.csv("data_norm2_pca.csv")
caffeNorm2.feature <-as.matrix(caffeNorm2.feature[,-1])

# SGD logistic regression
sgd_cvresult<-cvsgd.function(caffeNorm2.feature, label_train,5)

log.fit<-cv.glmnet(caffeNorm2.feature, label_train,family = "binomial", type.measure = "class",nfolds=5)
plot(log.fit)

# SVM
c=cvsvm.function(caffeNorm2.feature, label_train, 'radial',100,0.0001,5)

# GBM
c2 <- cv.function(caffeNorm2.feature, label_train, 9, 2000, 0.01, 5)


### CaffeConv3
caffeConv3.feature <- read.csv("data_conv3_pca.csv")
caffeConv3.feature <-as.matrix(caffeConv3.feature[,-1])
sgd_cvresult<-cvsgd.function(caffeConv3.feature, label_train,5)

# for only test
t<-sample(1:2000,1600)
train.data <- caffeConv3.feature[t,]
test.data <- caffeConv3.feature[-t,]
train.label <- matrix(label_train[t,])
test.label <- matrix(label_train[-t,])

# SGD logistic regression
sgd_cvresult<-cvsgd.function(train.data, train.label,5)

sgd_fit<- sgd(train.data, train.label,model='glm',model.control=binomial(link="logit"))
pred <- predict(sgd_fit, test.data,type = 'response')  
pred <- ifelse(pred <= 0.5, 0, 1) 
cv.error <- mean(pred != test.label)
print(cv.error)


### norm1 new
caffeNorm1new.feature <- read.csv("data_norm1_pca_new.csv")
caffeNorm1new.feature <-as.matrix(caffeNorm1new.feature[,-1])

# SGD logistic regression
sgd_cvresult<-cvsgd.function(caffeNorm1new.feature, label_train,5)


### CaffeConv4
caffeConv4.feature <- read.csv("data_conv4_pca.csv")
caffeConv4.feature <-as.matrix(caffeConv4.feature[,-1])
sgd_cvresult<-cvsgd.function(caffeConv4.feature, label_train,5)


### CaffeConv5
caffeConv5.feature <- read.csv("data_conv5_pca.csv")
caffeConv5.feature <-as.matrix(caffeConv5.feature[,-1])
sgd_cvresult<-cvsgd.function(caffeConv5.feature, label_train,5)


### CaffeFc8
caffeFc8.feature <- read.csv("data_fc8_pca.csv")
caffeFc8.feature <-as.matrix(caffeFc8.feature[,-1])
sgd_cvresult<-cvsgd.function(caffeFc8.feature, label_train,5)

### CaffeFc7 New
#logistic regression
caffeFc7New.feature <- read.csv("data_fc7_pca_new.csv")
caffeFc7New.feature <-as.matrix(caffeFc7New.feature[,-1])
sgd_cvresult<-cvsgd.function(caffeFc7New.feature, label_train,5)

### CaffeFc8 New
#logistic regression
caffeFc8New.feature <- read.csv("data_fc8_pca_new.csv")
caffeFc8New.feature <-as.matrix(caffeFc8New.feature[,-1])
sgd_cvresult<-cvsgd.function(caffeFc8New.feature, label_train,5)


#SVM
c=cvsvm.function(caffeFc7New.feature, label_train, 'radial',100,0.0001,5)



