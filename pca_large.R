##################################################
## Script for pca experiments on diagnosis dataset
##################################################
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

#preprocessing
# eliminate missing values
newBC<-BC_large
newBC<-newBC[complete.cases(newBC), ]

# transform the text into numeric dataframe
newBC<-sapply(newBC,as.numeric)
scaled<-as.data.frame(newBC)
scaled$diagnosis[scaled$diagnosis == 1] <- 0
scaled$diagnosis[scaled$diagnosis == 2] <- 1


# extract all positive and negative records
scaled_p<-subset(scaled,scaled$diagnosis==0)
scaled_n<-subset(scaled,scaled$diagnosis==1)
scaled_p.shuffle<-scaled_p[sample(nrow(scaled_p)),]
scaled_n.shuffle<-scaled_n[sample(nrow(scaled_n)),]

# split positive records evenly into three parts
row.p <-nrow(scaled_p.shuffle)
index.1<-round(0.33*row.p) #151
index.2<-round(0.66*row.p)  #307
split.1<-scaled_p.shuffle[1:(index.1+1),] #152
split.2<-scaled_p.shuffle[(index.1+2):(index.2+3),] #153
split.3<-scaled_p.shuffle[(index.2+4):row.p,] #153
list.p <- list(split.1, split.2, split.3)

# split negative records evenly into three parts
row.n <-nrow(scaled_n.shuffle)
index.1<-round(0.33*row.n) #80
index.2<-round(0.66*row.n)  #159
split.1<-scaled_n.shuffle[1:(index.1+1),] #81
split.2<-scaled_n.shuffle[(index.1+2):(index.2+2),] #80
split.3<-scaled_n.shuffle[(index.2+3):row.n,] #80
list.n <- list(split.1, split.2, split.3)

# combine positive and negative records into one group
splitall.1<-rbind(list.p[[1]],list.n[[1]])
splitall.2<-rbind(list.p[[2]],list.n[[2]])
splitall.3<-rbind(list.p[[3]],list.n[[3]])

# combine three groups 
train.list<-list(rbind(splitall.1,splitall.2),rbind(splitall.1,splitall.3),rbind(splitall.2,splitall.3))
test.list<-list(splitall.3,splitall.2,splitall.1)

#nn initialization
cv.error <- NULL
miss_classified <- NULL
miss_classified.rate<- NULL
k <- 3
pbar <- create_progress_bar('text')
pbar$init(k)

# cross valiation loops
for(i in 1:3){
  
  # extract one group of train and test data respectively
  train.cv<-train.list[[i]]
  test.cv<-test.list[[i]]
  
  #pca generation, the class is eliminated
  pca <- prcomp(train.cv[-1], scale. = T)
  
  #decide the number of pcas
  pca$rotation
  std_dev <-pca$sdev
  var <- std_dev^2
  prop_var <- var/sum(var)
  plot(prop_var, xlab = "Principal Component",
       ylab = "Single Proportion of Variance Explained",
       type = "b")
  plot(cumsum(prop_var), xlab = "Principal Component",
       ylab = "Cumulative Proportion of Variance Explained",
       type = "b")
  
  #combine the class attribute to the train set
  train.data <- data.frame(diagnosis = train.cv$diagnosis, pca$x)
  
  #select 19 PCAs out of 30 PCAs
  train.data <- train.data[,1:20]
  
  #perform the same PCA transformation on the test data (attribute of class is eliminated)
  test.data <- predict(pca, newdata = test.cv[-1])
  test.data <- as.data.frame(test.data)
  
  #select 19 PCAs out of 30 PCAs
  test.data <- test.data[,1:19]

  #nn train
  n<-names(train.data)
  f <- as.formula(paste("diagnosis ~", paste(n[!n %in% "diagnosis"], collapse = " + ")))
  nn <- neuralnet(f,data=train.data,hidden=c(11),linear.output=F)
  
  #nn test
  pr.nn <- neuralnet::compute(nn,test.data)
  pr.nn_ <- pr.nn$net.result
  print(pr.nn_)
  
  #test result transform
  pr.nn_[pr.nn_<0.5]<-0
  pr.nn_[pr.nn_>=0.5]<-1
  
  #error computation
  test.cv.r <- test.cv$diagnosis
  cv.error[i] <- sum((test.cv.r - pr.nn_)^2)/nrow(test.cv)
  miss_classified[i]<-length((pr.nn_-test.cv.r)[(pr.nn_-test.cv.r)!=0])

  #process show
  pbar$step()
}



