
# advance model for output on class
library(sgd)

label_train <- matrix(c(rep(1,1000),rep(0,1000)), ncol = 1)
label_testNew <- matrix(c(rep(1,1000),rep(0,1000)), ncol = 1)

train.data <- read.csv("data_fc7_pca.csv")
train.data <-as.matrix(train.data[,-1])

test.data <- read.csv("test_fc7_pca.csv")
test.data <-as.matrix(test.data[,-1])


sgd_fit<- sgd(train.data, label_train,model='glm',model.control=binomial(link="logit"))

pred <- predict(sgd_fit, test.data,type = 'response')  
pred <- ifelse(pred <= 0.5, 0, 1) 
cv.error <- mean(pred != label_testNew)
print(cv.error)

save(pred, file="./output/advance.test.pred.RData")
