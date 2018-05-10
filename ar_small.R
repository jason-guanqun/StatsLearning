##################################################
## Script for AR1 & AR2 experiments on original dataset
##################################################
library("arulesViz")
library(mlbench)
library(arules)
library(stargazer)

#load data
BC<-data("BreastCancer")
BC<-BreastCancer
summary(BC)

#preprocessing
newBC<-subset(BC,select = c(2:10))
newBC1<-subset(BC,select = c(2:11))
newBC2<-subset(newBC1,newBC1$Class=='malignant')[c(1:9)]
newBC3<-subset(newBC1,newBC1$Class=='benign')[c(1:9)]

#AR1
#large itemsets for class malignant
transaction <- as (newBC2, "transactions") 
itemsets <- apriori(transaction, parameter = list(target = "frequent itemsets"), control = list(verbose = FALSE))
result1 <- DATAFRAME(sort(itemsets, by="support"))
head(result1)

#large itemsets for class benign
transaction <- as (newBC3, "transactions") 
itemsets <- apriori(transaction, parameter = list(target = "frequent itemsets",minlen = 3), control = list(verbose = FALSE))
result2 <- DATAFRAME(sort(itemsets, by="support"))
head(result2)

#AR2
transaction <- as (newBC, "transactions") 
rules <- apriori(transaction, parameter = list(target = "rules"))
inspect(head(rules, by="lift"))


