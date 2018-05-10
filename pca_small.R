##################################################
## Script for pca experiments on original dataset
##################################################
library(mlbench)
library(neuralnet)
library(plyr)

#load data
BC<-data("BreastCancer")
BC<-BreastCancer

#preprocessing: clean -- select column, remove missing data, change it to be numeric dataframe

# we don't need the id number
newBC<-subset(BC,select = c(2:11))

# eliminate missing values
newBC<-newBC[complete.cases(newBC), ]

# transform the text into numeric dataframe
newBC<-sapply(newBC,as.numeric)
scaled<-as.data.frame(newBC)
scaled$Class[scaled$Class == 1] <- 0
scaled$Class[scaled$Class == 2] <- 1

# extract all positive and negative records
scaled_p<-subset(scaled,scaled$Class==0)
scaled_n<-subset(scaled,scaled$Class==1)
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
  pca <- prcomp(train.cv[1:9], scale. = T)
  
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
  train.data <- data.frame(Class = train.cv$Class, pca$x)
  
  #select 8 PCAs out of 9 PCAs
  train.data <- train.data[,-10]
  
  #perform the same PCA transformation on the test data (attribute of class is eliminated)
  test.data <- predict(pca, newdata = test.cv[1:9])
  test.data <- as.data.frame(test.data)
  
  #select 8 PCAs out of 9 PCAs
  test.data <- test.data[,-9]

  #nn train
  n<-names(train.data)
  f <- as.formula(paste("Class ~", paste(n[!n %in% "Class"], collapse = " + ")))
  nn <- neuralnet(f,data=train.data,hidden=c(11),linear.output=F,stepmax=15015)
 
  #nn test
  pr.nn <- neuralnet::compute(nn,test.data)
  pr.nn_ <- pr.nn$net.result
  print(pr.nn_)
  
  #test result transform
  pr.nn_[pr.nn_<0.5]<-0
  pr.nn_[pr.nn_>=0.5]<-1
  
  #error computation
  test.cv.r <- test.cv$Class
  cv.error[i] <- sum((test.cv.r - pr.nn_)^2)/nrow(test.cv)
  miss_classified[i]<-length((pr.nn_-test.cv.r)[(pr.nn_-test.cv.r)!=0])
  
  #process show
  pbar$step()
}



