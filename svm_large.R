##################################################
## Script for svm experiments on diagnosis dataset
##################################################

library(mlbench)
library(caret)
library(RCurl)
library(dplyr)

#load data
data_URL <- getURL('https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data')
names <- c('id_number', 'diagnosis', 'radius_mean', 
           'texture_mean', 'perimeter_mean', 'area_mean', 
           'smoothness_mean', 'compactness_mean', 
           'concavity_mean','concave_points_mean', 
           'symmetry_mean', 'fractal_dimension_mean',
           'radius_se', 'texture_se', 'perimeter_se', 
           'area_se', 'smoothness_se', 'compactness_se', 
           'concavity_se', 'concave_points_se', 
           'symmetry_se', 'fractal_dimension_se', 
           'radius_worst', 'texture_worst', 
           'perimeter_worst', 'area_worst', 
           'smoothness_worst', 'compactness_worst', 
           'concavity_worst', 'concave_points_worst', 
           'symmetry_worst', 'fractal_dimension_worst')
BC_large <- read.table(textConnection(data_URL), sep = ',', col.names = names)

# we don't need id_numer as input
BC_large$id_number <- NULL

newBC<-BC_large #all features
#newBC<-subset(BC_large,select = c(diagnosis,radius_worst,area_worst,radius_mean, perimeter_mean,area_mean)) #5 features: AR2
#newBC<-subset(BC_large,select = -area_mean) #features without area_mean: AR1

# eliminate missing values
newBC<-newBC[complete.cases(newBC), ]

# change the Class text into numeric
newBC<-sapply(newBC,as.numeric)
newBC<-as.data.frame(newBC)
newBC$diagnosis[newBC$diagnosis == 1] <- 0
newBC$diagnosis[newBC$diagnosis == 2] <- 1

#set the class data as factor
newBC[["diagnosis"]] = factor(newBC[["diagnosis"]])

#set train control - 3-fold cross validation
trctrl <- trainControl(method = "cv", number = 3)

#svm
svm_Linear <- train(diagnosis ~., data = newBC, method = "svmLinear",
                    trControl=trctrl,
                    preProcess = c("center", "scale"),
                    tuneLength = 10)
#show results
svm_Linear 




