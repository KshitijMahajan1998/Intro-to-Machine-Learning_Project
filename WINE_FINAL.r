#Problem Statement: Predicting the Quality of white wine on a scale of 1-10 based on other predictor variables.
#(Regression Problem)

#Group Members:
#1. Meeth Yogesh Handa (EID:mh58668)
#2. Audrey Hsien (EID: arh4247)
#3. Kshitij Mahajan (EID: ksm3267)
#4. Anthony Moreno (EID: am83596)
#5. Milan Patel (EID: mp47736)
#6. Varun Kausika (EID: vsk394)


#Note: The RMSE values mentioned are for reference.
#Note: It is assumed that data csv file is in working directory.
#Individual values may vary according to train/test data split.

datawq = read.csv('winequality-white.csv', header=TRUE)
View(datawq)
attach(datawq)
library(Boruta)
library(tree)
library(gbm) #boost package
library(randomForest) 
library(MASS)
library(readr)
library(kknn)


hist(datawq$quality)

boruta_output = Boruta(quality~., data=na.omit(datawq), doTrace=2)  
boruta_signif = names(boruta_output$finalDecision[boruta_output$finalDecision %in% c("Confirmed", "Tentative")])  # collect Confirmed and Tentative variables
print(boruta_signif)
plot(boruta_output, cex.axis=.7, las=2, xlab="", main="Variable Importance")
#3 most important variables as per Boruta: volatile.acidity, alcohol, free.sulfur.dioxide

set.seed(1)
# Random sampling
#KNN
samplesize = 0.50 * nrow(datawq)
index = sample( seq_len ( nrow ( datawq ) ), size = samplesize )

# Create training and test set
train = datawq[ index, ]
test = datawq[ -index, ]

ind = order(test[,1])
test =test[ind,]

MSE = NULL

kk = c(2,10,50,100,150,200,250,300,400,505)

for(i in kk){
  
  near = kknn(quality~.,train,test,k=i,kernel = "rectangular")
  aux = mean((test[,2]-near$fitted)^2)
  
  MSE = c(MSE,aux)
  
  plot(quality,main=paste("k=",i),pch=19,cex=0.8,col="darkgray")
  lines(test[,1],near$fitted,col=2,lwd=2)
  cat ("Press [enter] to continue")
  line <- readline()
}


plot(log(1/kk),sqrt(MSE),type="b",xlab="Complexity (log(1/k))",col="blue",ylab="RMSE",lwd=2,cex.lab=1.2)
text(log(1/kk[1]),sqrt(MSE[1])+0.3,paste("k=",kk[1]),col=2,cex=1.2)
text(log(1/kk[10])+0.4,sqrt(MSE[10]),paste("k=",kk[10]),col=2,cex=1.2)
text(log(1/kk[5])+0.4,sqrt(MSE[5]),paste("k=",kk[5]),col=2,cex=1.2)


set.seed(1)


#Fitting regression tree to training data set:
train = sample(1:nrow(datawq),nrow(datawq)/2)
tree.winequality = tree(quality~.,data=datawq,subset=train)
summary(tree.winequality)
plot(tree.winequality)
text(tree.winequality,pretty=0)

#The training model is now implemented on test data and RMSE is calculated:
yhat = predict(tree.winequality,newdata=datawq[-train,])
datawq.test = datawq[-train,"quality"]
plot(yhat,datawq.test)
abline(0,1)
sqrt(mean((yhat-datawq.test)^2))
#RMSE = 0.75

#Trying randomForest on training data to see if error reduces further:
rf.winequality = randomForest(quality~.,data=datawq,susbet=train,mtry=4,importance=T)
rf.winequality
plot(rf.winequality)
summary(rf.winequality)
#RMSE reduces further on implementing random forest


#Applying random forest on test data:
yhat.rf = predict(rf.winequality,datawq[-train,])
plot(yhat.rf,datawq.test)
abline(0,1)
sqrt(mean((yhat.rf-datawq.test)^2))
#RMSE = 0.25
#Fantastic improvement - the MSE reduces massively on test data after random forest
importance(rf.winequality)
varImpPlot(rf.winequality)


#Bagging implemented on training data set to see the change in results:
bag.winequality = randomForest(quality~.,data=datawq,subset=train,mtry=11,importance=T)
bag.winequality
summary(bag.winequality)
plot(bag.winequality)
#Finding: RMSE reduces considerably after applying bagging compared to the regression tree tried before

#Finding out most important variables in bagging approach:
importance(bag.winequality)
#Finding: The 3 most important variables are the same as before:
#1. alcohol
#2. volatile.acidity
#3. free.sulfur.dioxide

#Using bagging approach on test data:
yhat.bag = predict(bag.winequality,datawq[-train,])
plot(yhat.bag,datawq.test)
abline(0,1)
sqrt(mean((yhat.bag-datawq.test)^2))
#RMSE = 0.63
#RMSE reduces for test data too after bagging


#Implementing bossting on training data:
boost.winequality = gbm(quality~.,data=datawq[train,],distribution="gaussian",n.trees=1000,interaction.depth=2,shrinkage=0.1)
summary(boost.winequality)

#Applying boosting on test data:
yhat.boost = predict(boost.winequality,newdata=datawq[-train,],ntrees=1000,interaction.depth=2,shrinkage=0.1)
sqrt(mean((yhat.boost-datawq.test)^2))
#RMSE = 0.69
#Test MSE is still the best for random forest algorithm so far

#Changing boosting parameters
boost.winequality2 = gbm(quality~.,data=datawq[train,],distribution="gaussian",n.trees=5000,interaction.depth=2,shrinkage=0.02)
yhat.boost = predict(boost.winequality2,newdata=datawq[-train,],ntrees=5000,interaction.depth=2,shrinkage=0.02)
sqrt(mean((yhat.boost-datawq.test)^2))
#RMSE = 0.68
#Test MSE is still the best for random forest algorithm so far


library(xgboost)
training.x = model.matrix(quality~., data = datawq[train,])
testing.x = model.matrix(quality~., data = datawq[-train,])

model.XGB = xgboost(data = data.matrix(training.x[,-1]),
                    label = datawq[train,]$quality,
                    eta = 0.1,
                    max_depth =20,
                    nrounds = 50,
                    objective = 'reg:linear')

# Train RMSE = 0.078

# Obtaining test error
y_pred = predict(model.XGB, data.matrix(testing.x[,-1]))
RMSE = mean((y_pred - datawq.test)^2)
print(RMSE)
# Test RMSE = 0.45

#Doing forward stepwise selection to pick most effective variables for regression
library(leaps)
regfit.fwd=regsubsets(quality~.,data=datawq,nvmax=10,method ="forward")
summary(regfit.fwd)
#Best variable for 1 variable model: alcohol
#Best variables for a multiple linear regression model: alcohol, volatile.acidity, residual.sugar

#Implementing simple linear regression on training data set:
lm.winequality = lm(quality~alcohol,data=datawq[train,])
lm.winequality
summary(lm.winequality)

#Implementing simple linear regression on test data set:
yhat.reg = predict(lm.winequality,data=datawq,susbset=datawq[-train,])
summary(yhat.reg)
sqrt(mean((yhat.reg-datawq.test)^2))
#Test error RMSE = 0.93


#Implementing multiple linear regression using the 3 most important parameters on training data set:
lm.winequality2 = lm(quality~alcohol+volatile.acidity+residual.sugar,data=datawq[train,])
lm.winequality2
summary(lm.winequality2)

#Implementing multiple linear regression using the 3 most important parameters on test data set:
yhat.mreg = predict(lm.winequality2,data=datawq,susbset=datawq[-train,])
summary(yhat.mreg)
sqrt(mean((yhat.mreg-datawq.test)^2))
#Test error RMSE = 0.96

#Neural net:
#getwd()
#data <- read.csv("winequality-white.csv")
str(datawq)
print(datawq)
# Random sampling
samplesize = 0.50 * nrow(datawq)
set.seed(1)
index = sample( seq_len ( nrow ( datawq ) ), size = samplesize )

# Create training and test set
datatrain = datawq[ index, ]
datatest = datawq[ -index, ]
summary(datatrain)
summary(datatest)

## Scale/Standardized data for neural network

max = apply(datawq , 2 , max)
min = apply(datawq, 2 , min)
scaled = as.data.frame(scale(datawq, center = min, scale = max - min))

# load library
library(neuralnet)

# creating training and test set
trainNN = scaled[index , ]
testNN = scaled[-index , ]

# fit neural network
#Note: The neural net takes some time to build - please wait
set.seed(1)
NN = neuralnet(quality~., trainNN, hidden = 7 , linear.output = T,stepmax = 1e7 )

# plot neural network
plot(NN)
## Prediction using neural network

predict_testNN = compute(NN, testNN[,c(1:11)])
predict_testNN = (predict_testNN$net.result * (max(datawq$quality) - min(datawq$quality))) + min(datawq$quality)
plot(datatest$quality, predict_testNN, col='blue', pch=16, ylab = "predicted quality NN", xlab = "real quality")
abline(0,1)

# Calculate Root Mean Square Error (RMSE)
RMSE.NN = (sum((datatest$quality - predict_testNN)^2) / nrow(datatest)) ^ 0.5 
RMSE.NN
#RMSE = 0.71

#Results: The test RMSE value was lowest for Random Forest algorithm.
#We have successfully predicted the quality of white wine on a scale of 1-10 (with decimal value precision)