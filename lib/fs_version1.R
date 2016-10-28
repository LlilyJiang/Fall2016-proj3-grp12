
source(train.r)
source(test.r)

###### First part : testing start code using all 5000 SIFT features
# read data
sift.feature=read.csv("/Users/Amyummy/Documents/Rstudio/ads_pro3/Project3_poodleKFC_train/sift_features.csv")
dim(sift.feature)
sift=t(sift.feature)

# add lables: 0 for dog and 1 for fried chicken
label1=append(rep(1,1000),rep(0,1000))
data=cbind(sift,label1)
dim(data)

# divide data to training part(1400 images) and testing part(600 images).
set.seed(0)
sam=sample(1:2000,1400)

train_data0=data[sam,]
dat_train=train_data0[,1:5000]
label_train=train_data0[,5001]

test_data0=data[-sam,]
dat_test=test_data0[,1:5000]
label_test=test_data0[,5001]
dim(dat_train)
dim(dat_test)

# run train.r and test.r
fit_train=train(dat_train,label_train)
pred=test(fit_train, dat_test)

# use the prediction to calculte accuracy
x=label_test-pred
k=0
for(i in 1:600){
  if(x[i]==0){
    k=k+1
  }else{k=k}
}
accu=k/600
accu
# one time accuracy 70%


################## Part2: SIFT feature selection
# three methods:
# 1. based on variance(cut off)
# 2. PCA
# 3. Random forest to choose important features

#############################     method1: based on variance     #############################
set.seed(1)
vari=rep(0,5000)
for(i in 1:5000){
  vari[i]=var(data[,i])
}
max(vari)
min(vari)

# cut value = 0.5e-6
n=0
for(i in 1:5000){
  if(vari[i]>=0.5e-6){n=n+1}
  else{n=n}
}
n
# if variance > 0.5e-6, we keep the feature, then there are 1968(n) features remaining
cut=rep(0.5e-6,5000)
getcol=vari-cut>=0
data.m1=data[,getcol]
dim(data.m1)

# data.m1 is the output of feature selection method1

#new feature set selected by method1
#set.seed()
sam1=sample(1:2000,1400)
train_data1=data.m1[sam1,]
dat_train1=train_data1[,1:1968]
label_train1=train_data1[,1969]

test_data1=data.m1[-sam1,]
dat_test1=test_data1[,1:1968]
label_test1=test_data1[,1969]

# run train.r and test.r
install.packages("gbm")
library(gbm)
fit_train1=train(dat_train1,label_train1)
pred1=test(fit_train1, dat_test1)

# use the prediction to calculte accuracy
x1=label_test1-pred1
k=0
for(i in 1:600){
  if(x1[i]==0){
    k=k+1
  }else{k=k}
}
accu1=k/600
accu1
# result 66 %. costs about 3 minutes

#############################     end of method1     #############################

#############################     method2: PCA     #############################
# dim(data)= 2000*5001
set.seed(2)
features=data[,1:5000]; 
# princomp() can only be used with more units than variables
# use prcomp()
pca=prcomp(features,center=TRUE, scale=TRUE);

# Cum.Screeplot.
pr_var=(pca$sdev)^2;
plot(cumsum(pr_var)/sum(pr_var)*100,ylim=c(0,100),type="b",xlab="component",ylab="c umulative propotion (%)",main="Cum. Scree plot");
abline(h=70,col="red");
plot(pca)

# considering accuracy and time trade-off, we keep first 500 pcs

# The rotation measure provides the principal component loading
# for score, we use the first 500 pcs to replace 5000 features
n2=500
dim(pca$x[,1:500])
data.m2=cbind(pca$x[,1:500],label1)
dim(data.m2)
# 
#new feature set selected by method1
sam2=sample(1:2000,1400)
train_data2=data.m2[sam2,]
dat_train2=train_data2[,1:500]
label_train2=train_data2[,501]

test_data2=data.m2[-sam2,]
dat_test2=test_data2[,1:500]
label_test2=test_data2[,501]

# run train.r and test.r
fit_train2=train(dat_train2,label_train2)
pred2=test(fit_train2, dat_test2)

# use the prediction to calculte accuracy
x2=label_test2-pred2
k=0
for(i in 1:600){
  if(x2[i]==0){
    k=k+1
  }else{k=k}
}
accu2=k/600
accu2
# result 61.5%. costs about 1 minute

#############################     end of method2     #############################


#############################     method3: random forest     ################################
# since the permutation variable importance is affected by collinearity
# it's necessary to handle collinearity prior to running random forest for extracting important variables.

########### 1.deal with collinearity
# sessionInfo(): R version 3.2.2 (caret only available for version > 3.2.5)

# load required libraries
install.packages("caret", dependencies = c("Depends", "Suggests"))
library(ggplot2)
library(caret)
install.packages("corrplot")
library(corrplot)
library(plyr)

# give each feature a "name" and Calculate correlation matrix
feature1=features
colnames(feature1)[1:5000]=as.character(seq(1,5000,by=1))
descrCor=cor(feature1)

# Print correlation matrix and look at max correlation
summary(descrCor[upper.tri(descrCor)])

# Check Correlation Plot
#corrplot(descrCor, order = "FPC", method = "color", type = "lower", tl.cex = 0.7, tl.col = rgb(0, 0, 0))

# find attributes that are highly corrected
highlyCorrelated=findCorrelation(descrCor, cutoff=0.6)

# print indexes of highly correlated attributes
print(highlyCorrelated)

# Indentifying Variable Names of Highly Correlated Variables
highlyCorCol=colnames(feature1)[highlyCorrelated]

# Print highly correlated attributes
highlyCorCol

# Remove highly correlated variables and create a new dataset
features1=feature1[, -which(colnames(feature1) %in% highlyCorCol)]
dim(features1)
# after remove highly corelated variables, there are still 4913 features remaining.


########### 2.Use random forest
# ensure the results are repeatable
set.seed(3)
install.packages("randomForest")
library(randomForest)

df=as.data.frame(cbind(features1,label1))
allX=paste("X",1:ncol(features1),sep="")
names(df)=c(allX,"label")


#Train Random Forest
rf=randomForest(as.factor(label)~.,data=df, importance=TRUE,ntree=500)
# it takea about 10 minutes

#Evaluate variable importance
imp = importance(rf, type=1)
imp = data.frame(predictors=rownames(imp),imp)

# Order the predictor levels by importance
imp.sort <- arrange(imp,desc(MeanDecreaseAccuracy))
imp.sort$predictors <- factor(imp.sort$predictors,levels=imp.sort$predictors)

# Select the top 100 predictors
imp.100=imp.sort[1:100,]
print(imp.100)

# Plot Important Variables
varImpPlot(rf, type=1)

# Subset data with 100 independent and 1 dependent variables
data.m3 = cbind(df[,c(imp.100$predictors)],lable = df$label)

# new feature set selected by method3
sam3=sample(1:2000,1400)
train_data3=data.m3[sam3,]
dat_train3=train_data3[,1:100]
label_train3=train_data3[,101]

test_data3=data.m3[-sam3,]
dat_test3=test_data3[,1:100]
label_test3=test_data3[,101]

# run train.r and test.r
fit_train3=train(dat_train3,label_train3)
pred3=test(fit_train3, dat_test3)

# use the prediction to calculte accuracy
x3=label_test3-pred3
k=0
for(i in 1:600){
  if(x3[i]==0){
    k=k+1
  }else{k=k}
}
accu3=k/600
accu3
# result 60%. costs about 10 seconds


#############################     end of method3     #############################


