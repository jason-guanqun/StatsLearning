##############################################################
## Script for neural network experiments on diagnosis dataset
###############################################################
library(mlbench)
library(neuralnet)
library(plyr)
library(RCurl)
library(dplyr)

#get data and summary
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


data<-BC_large #all features
#data<-subset(BC_large,select = c(diagnosis,radius_worst,area_worst,radius_mean, perimeter_mean,area_mean)) #5 features: AR2
#data<-subset(BC_large,select = -area_mean) #features without area_mean: AR1

# eliminate missing values
data<-data[complete.cases(data), ]

# change the Class text into numeric
data<-sapply(data,as.numeric)
data<-as.data.frame(data)
data$diagnosis[data$diagnosis == 1] <- 0
data$diagnosis[data$diagnosis == 2] <- 1

#nn initialization
n<-names(data) 
cv.error <- NULL
miss_classified <- NULL
k <- 3
pbar <- create_progress_bar('text')
pbar$init(k)


f <- as.formula(paste("diagnosis ~", paste(n[!n %in% "diagnosis"], collapse = " + ")))

# extract all positive and negative records
data_p<-subset(data,data$diagnosis==0)
data_n<-subset(data,data$diagnosis==1)
data_p.shuffle<-data_p[sample(nrow(data_p)),]
data_n.shuffle<-data_n[sample(nrow(data_n)),]

# split positive records evenly into three parts
row.p <-nrow(data_p.shuffle)
index.1<-round(0.33*row.p)
index.2<-round(0.66*row.p)
split.1<-data_p.shuffle[1:(index.1+1),] 
split.2<-data_p.shuffle[(index.1+2):(index.2+3),] 
split.3<-data_p.shuffle[(index.2+4):row.p,]
list.p <- list(split.1, split.2, split.3)

# split negative records evenly into three parts
row.n <-nrow(data_n.shuffle)
index.1<-round(0.33*row.n) 
index.2<-round(0.66*row.n)  
split.1<-data_n.shuffle[1:(index.1+1),] 
split.2<-data_n.shuffle[(index.1+2):(index.2+2),] 
split.3<-data_n.shuffle[(index.2+3):row.n,]
list.n <- list(split.1, split.2, split.3)

# combine positive and negative records into one group
splitall.1<-rbind(list.p[[1]],list.n[[1]])
splitall.2<-rbind(list.p[[2]],list.n[[2]])
splitall.3<-rbind(list.p[[3]],list.n[[3]])

# combine three groups
train.list<-list(rbind(splitall.1,splitall.2),rbind(splitall.1,splitall.3),rbind(splitall.2,splitall.3))
test.list<-list(splitall.3,splitall.2,splitall.1)

# cross valiation loops
for(i in 1:3){
  train.cv<-train.list[[i]]
  test.cv<-test.list[[i]]
  
  # data normalization, class attribute is eliminated
  train.scale<-as.data.frame(scale(train.cv[-1]))
  test.scale<-as.data.frame(scale(test.cv[-1]))
  
  # add the class attribute
  train.scale$diagnosis = train.cv$diagnosis
  test.scale$diagnosis = test.cv$diagnosis
  
  #nn train
  nn <- neuralnet(f,data=train.scale,hidden=c(11),linear.output=F)
  
  #nn test
  pr.nn <- neuralnet::compute(nn,test.scale[,-ncol(test.scale)])
  pr.nn_ <- pr.nn$net.result
  print(pr.nn_)
  
  #test result transform
  pr.nn_[pr.nn_<0.5]<-0
  pr.nn_[pr.nn_>=0.5]<-1
  
  #error computation
  test.scale.r <- test.scale$diagnosis
  cv.error[i] <- sum((test.scale.r - pr.nn_)^2)/nrow(test.scale)
  miss_classified[i]<-length((pr.nn_-test.scale.r)[(pr.nn_-test.scale.r)!=0])
  pbar$step()
}

