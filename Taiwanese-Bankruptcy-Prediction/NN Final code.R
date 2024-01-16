# Install and load necessary packages
install.packages("keras")
install.packages("reticulate")
install.packages("tensorflow")
library(tensorflow)
install_tensorflow()
install.packages("neuralnet")
library(neuralnet)

install.packages("DMwR2")
install.packages("DMwR")
install.packages("curl")
install.packages("hms")
install.packages("TTR")
install.packages("ROSE")
library(ROSE)

library(DMwR2)
library(DMwR)

library(keras)
library(caret)
library(ggplot2)
library(lattice)
library(dplyr)
library(proxy)



library(tidyverse)
library(dplyr)
library(caret)
library(psych)
library(fastDummies)
library(nnet)
library(e1071)
library(kernlab)
library(randomForest)
library(ggplot2)
library(nortest)
library(performanceEstimation)
library(rpart)
library(rpart.plot)
library(reshape)
library(adabag)
library(mboost)
library(neuralnet)
library(gbm)
library(cluster)
library(factoextra)



data <- read.csv("NN-Data.csv")
data
str(data)


# Split the data into training and testing sets
set.seed(123)
split_index <- createDataPartition(data$Bankrupt, p = 0.6, list = FALSE)
train_data <- data[split_index, ]
str(train_data)
test_data <- data[-split_index, ]
test_data
#train_data_smote <- ovun.sample(Bankrupt ~ ., data = train_data, method = "over", N = 7000)$data
#test_data_smote <- ovun.sample(Bankrupt ~ ., data = test_data, method = "over", N = 7000)$data
smote_train <- smote(Bankrupt ~ ., data = train_data)
smote_test <- smote(Bankrupt ~ ., data = test_data)
table(smote_train$Bankrupt)




#Split trial 2
# Assuming 'your_data' is your original dataset
set.seed(123)  # for reproducibility
indices <- createDataPartition(data$Bankrupt, p = 0.7, list = FALSE)

# Create training and initial test sets
train_data_initial <- data[indices, ]
test_data_initial <- data[-indices, ]


# Manually balance the test set
n <- min(sum(train_data_initial$Bankrupt == 0), sum(train_data_initial$Bankrupt == 1))
train_data_balanced <- rbind(
  na.omit(train_data_initial[train_data_initial$Bankrupt == 0, ])[1:n, ],
  na.omit(train_data_initial[train_data_initial$Bankrupt == 1, ])[1:n, ]
)


# Manually balance the test set
n <- min(sum(test_data_initial$Bankrupt == 0), sum(test_data_initial$Bankrupt == 1))
test_data_balanced <- rbind(
  na.omit(test_data_initial[test_data_initial$Bankrupt == 0, ])[1:n, ],
  na.omit(test_data_initial[test_data_initial$Bankrupt == 1, ])[1:n, ]
)



#Neural Network code
nn <- neuralnet(Bankrupt ~ ., data = train_data, linear.output = F, hidden = 2, stepmax = 1e6)
plot(nn, rep="best")

nn.pred <- predict(nn, train_data_balanced, type = "response")
nn.pred.classes <- ifelse(nn.pred > 0.5, 1, 0)
Conf_mat_train <- confusionMatrix(as.factor(nn.pred.classes), as.factor(train_data_balanced$Bankrupt))
Conf_mat_train

nn.pred <- predict(nn, test_data_balanced, type = "response")
nn.pred.classes <- ifelse(nn.pred > 0.5, 1, 0)
Conf_mat_test <- confusionMatrix(as.factor(nn.pred.classes), as.factor(test_data_balanced$Bankrupt))
Conf_mat_test
