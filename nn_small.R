##################################################
## Script for NN experiments on original dataset
##################################################
library("mlbench")
library(neuralnet)
library(plyr)
data("BreastCancer")

# feature selection
# data<-BreastCancer[,c(2:11)]
# data<-BreastCancer[,c(2,6,8,9,11)]
data<-BreastCancer[,c(-1,-3)]

# check data
data<-data[complete.cases(data), ]
data<-sapply(data,as.numeric)
data<-as.data.frame(data)

# for small dataset assign classes
data$Class[data$Class == 1] <- 0
data$Class[data$Class == 2] <- 1

n<-names(data) 

f <- as.formula(paste("Class ~", paste(n[!n %in% "Class"], collapse = " + ")))

data_p<-subset(data,data$Class==0)
data_n<-subset(data,data$Class==1)

data_p.shuffle<-data_p[sample(nrow(data_p)),]
data_n.shuffle<-data_n[sample(nrow(data_n)),]

# extract all positive and negative records
row.p <-nrow(data_p.shuffle)
index.1<-round(0.33*row.p) #151
index.2<-round(0.66*row.p)  #307
split.1<-data_p.shuffle[1:(index.1+1),] #152
split.2<-data_p.shuffle[(index.1+2):(index.2+3),] #153
split.3<-data_p.shuffle[(index.2+4):row.p,] #153
list.p <- list(split.1, split.2, split.3)

# split negative records evenly into three parts
row.n <-nrow(data_n.shuffle)
index.1<-round(0.33*row.n) #80
index.2<-round(0.66*row.n)  #159
split.1<-data_n.shuffle[1:(index.1+1),] #81
split.2<-data_n.shuffle[(index.1+2):(index.2+2),] #80
split.3<-data_n.shuffle[(index.2+3):row.n,] #80
list.n <- list(split.1, split.2, split.3)

splitall.1<-rbind(list.p[[1]],list.n[[1]])
splitall.2<-rbind(list.p[[2]],list.n[[2]])
splitall.3<-rbind(list.p[[3]],list.n[[3]])

# combine three groups 
train.list<-list(rbind(splitall.1,splitall.2),rbind(splitall.1,splitall.3),rbind(splitall.2,splitall.3))
test.list<-list(splitall.3,splitall.2,splitall.1)

# parameter normalization
cv.error <- NULL
miss_classified <- NULL
k <- 3

# cross valiation loops'
for(i in 1:3){
  # extract one group of training and test data
  train.cv<-train.list[[i]]
  test.cv<-test.list[[i]]
  
  train.scale<-as.data.frame(scale(train.cv[-10]))
  test.scale<-as.data.frame(scale(test.cv[-10]))
  
  train.scale$Class = train.cv$Class
  test.scale$Class = test.cv$Class

  # train neural network model (33 epoch)
  nn <- neuralnet(f,data=train.scale,hidden=c(11),linear.output=F, stepmax = 15015)
  # predict on test data
  pr.nn <- neuralnet::compute(nn,test.scale[,-ncol(test.scale)])
  
  pr.nn_ <- pr.nn$net.result
  print(pr.nn_)
  test.scale.r <- test.scale$Class
  # test.scale.r <- test.scale$diagnosis
  pr.nn_[pr.nn_<0.5]<-0
  pr.nn_[pr.nn_>=0.5]<-1
  
  # compute cross validation error
  cv.error[i] <- sum((test.scale.r - pr.nn_)^2)/nrow(test.scale)
  miss_classified[i]<-length((pr.nn_-test.scale.r)[(pr.nn_-test.scale.r)!=0])
}

