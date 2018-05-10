##################################################
## Script for AR1 & AR2 experiments on original dataset
##################################################
library(RCurl)
library(dplyr)
library(arulesViz)
library(arules)

#get data and summary
data_URL <-
  getURL(
    'https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data'
  )
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
BC <- read.table(textConnection(data_URL), sep = ',', col.names = names)
summary(BC)

# we don't need id_numer as input
BC$id_number <- NULL

# data should be discretized first for the next transaction generation process
for (i in 2:31) {
  BC[, i] <- discretize(BC[, i])
}


#data preprocessing
newBC <- subset(BC, select = -1)
newBC1 <- subset(BC, BC$diagnosis == 'M')
newBC1 <- subset(newBC1, select = -1)
newBC2 <- subset(BC, BC$diagnosis == 'B')
newBC2 <- subset(newBC2, select = -1)

#AR1
#large itemsets for class malignant
transaction <- as (newBC1, "transactions") 
itemsets <- apriori(transaction, parameter = list(target = "frequent itemsets", minlen = 5), control = list(verbose = FALSE))
inspect(head(itemsets, by = "support"))

#large itemsets for class benign
transaction <- as (newBC2, "transactions") 
itemsets <- apriori(transaction, parameter = list(target = "frequent itemsets",minlen = 5), control = list(verbose = FALSE))
inspect(head(itemsets, by = "support"))

#AR2
transaction <- as (newBC, "transactions")
rules <- apriori(transaction, parameter = list(target = "rules"))
inspect(head(rules, by = "lift"))




