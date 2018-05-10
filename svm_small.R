##################################################
## Script for svm experiments on original dataset
##################################################
library(mlbench)
library(caret)

#load data
BC<-data("BreastCancer")
BC<-BreastCancer
newBC<-BreastCancer[,c(2,6,8,9,11)] #4 features:AR2
# newBC<-subset(BC,select = c(-1,-3)) #8 features:AR1
# newBC<-subset(BC,select = c(2:11)) #9 features

# eliminate missing values
newBC<-newBC[complete.cases(newBC), ]

# change the Class text into numeric
newBC<-sapply(newBC,as.numeric)
newBC<-as.data.frame(newBC)
newBC$Class[newBC$Class == 1] <- 0
newBC$Class[newBC$Class == 2] <- 1

#set the class data as factor
newBC[["Class"]] = factor(newBC[["Class"]])

#set train control - 3-fold cross validation
trctrl <- trainControl(method = "cv", number = 3)

#svm
svm_Linear <- train(Class ~., data = newBC, method = "svmLinear",
                    trControl=trctrl,
                    preProcess = c("center", "scale"),
                    tuneLength = 10)
#show results
svm_Linear 




