#part=list(depth=7,Ntrees=900,Shrinkage=0.5)


# This is used for calculating accuracy
# Input is:
# 1.par_best after cross validation
# 2.dat_train, label_train used to train the model in train.r
# 3.label_test and dat_test used to classification and get accuracy



#dat_train=
#label_train=
fit_train=train(dat_train,label_train,par_best)

#dat_test=
pred=test(fit_train, dat_test)

x=label_test-pred
k=0
for(i in 1:600){
     if(x[i]==0){
         k=k+1
       }else{k=k}
   }
accu=k/600
accu

