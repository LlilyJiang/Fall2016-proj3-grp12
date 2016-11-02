############## test error for 10 layers ########
#### logistic regression ###
## 1600 train + cv
## 400 test
### caffeNorm1


caffeNorm1.feature <- read.csv("data_norm1_pca.csv")
caffeNorm1.feature <-as.matrix(caffeNorm1.feature[,-1])

### caffeNorm2
caffeNorm2.feature <- read.csv("data_norm2_pca.csv")
caffeNorm2.feature <-as.matrix(caffeNorm2.feature[,-1])

### caffeConv3
caffeConv3.feature <- read.csv("data_conv3_pca.csv")
caffeConv3.feature <-as.matrix(caffeConv3.feature[,-1])

### caffeConv4
caffeConv4.feature <- read.csv("data_conv4_pca.csv")
caffeConv4.feature <-as.matrix(caffeConv4.feature[,-1])

### caffeConv5
caffeConv5.feature <- read.csv("data_conv5_pca.csv")
caffeConv5.feature <-as.matrix(caffeConv5.feature[,-1])

### caffeConv5
caffePool5.feature <- read.csv("data_pool5_pca.csv")
caffePool5.feature <-as.matrix(caffePool5.feature[,-1])

### caffeFc6
caffeFc6.feature <- read.csv("data_fc6_pca.csv")
caffeFc6.feature <-as.matrix(caffeFc6.feature[,-1])

### caffeFc7
caffeFc7.feature <- read.csv("data_fc7_pca.csv")
caffeFc7.feature <-as.matrix(caffeFc7.feature[,-1])

### caffeFc8
caffeFc8.feature <- read.csv("data_fc8_pca.csv")
caffeFc8.feature <-as.matrix(caffeFc8.feature[,-1])

###Prob
caffeProb.feature <- read.csv("data_prob_pca.csv")
caffeProb.feature <-as.matrix(caffeProb.feature[,-1])


#Model
library(sgd)
t<-sample(1:2000,1600)
# can custermize
data_train<-caffePool5.feature
train.data <- data_train[t,]
test.data <- data_train[-t,]
train.label <- matrix(label_train[t,])
test.label <- matrix(label_train[-t,])


sgd_fit<- sgd(train.data, train.label,model='glm',model.control=binomial(link="logit"))
pred <- predict(sgd_fit, test.data,type = 'response')  
pred <- ifelse(pred <= 0.5, 0, 1) 
cv.error <- mean(pred != test.label)
print(cv.error)






