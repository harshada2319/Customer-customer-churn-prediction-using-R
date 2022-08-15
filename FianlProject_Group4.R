# GROUP 4
##CUSTOMER CHURN PREDICTION##

#Clear workspace
rm(list=ls())

#Set Directory 
setwd("C:/Users/harsh/OneDrive/Documents/MSBALeBow/quarter2/Data Mining Dr Hill")

# Install new packages
install.packages(c("ggplot2",
                   "caret"))
install.packages(c("rpart",
                   "rpart.plot"))
install.packages("randomForest")
install.packages("caretEnsemble")

# Load libraries

library(ggplot2)
library(caret)
library(rpart)
library(rpart.plot)
library(randomForest)
library(caretEnsemble)

# Load data file
customer <- read.csv(("CustomerChurn.csv"),
                     stringsAsFactors = FALSE)

#Data Overview

#Preview
head(x = customer)
tail(x = customer)

str(customer)

# We can identify rows with missing values 
#sapply(customer, function(x) sum(is.na(x)))
customer[!complete.cases(customer), ]

Na_rows <- rownames(customer)[!complete.cases(customer)]
Na_rows

#We take only the complete cases
customer <- customer[complete.cases(customer), ]
any(is.na(customer))


#Identify variables types and create convenience vectors
fac <- c("gender","SeniorCitizen","Partner","Dependents","PhoneService","MultipleLines","InternetService",
         "OnlineSecurity","OnlineBackup","DeviceProtection","TechSupport","StreamingTV","StreamingMovies",
         "Contract","PaperlessBilling","PaymentMethod")
#num <- c("MonthlyCharges","TotalCharges","tenure")
num <- c("tenure","MonthlyCharges","TotalCharges")
allvar <- c(fac,num)
customer$Churn <- factor(customer$Churn)

summary(object = customer$tenure)

#we want to see all the unique values the features can take
lapply(X = customer[ ,fac], 
       FUN = unique)

#converting to factors
customer[ ,fac] <- lapply(X = customer[ , fac], 
                          FUN = factor)


#Bar plot for our target variable
par(mar = c(2, 2, 2, 2))
plot(customer$Churn,
     main = "Churn")

#Correlation plot for our numeric variables
co <- cor(x = customer[ ,c("MonthlyCharges","TotalCharges","tenure")], y = NULL, use = "everything",
          method = c("pearson", "kendall", "spearman"))
co
#par(mar = c(2, 2, 2, 0))
corrplot::corrplot(co, method="number")

#We take out totalCharges to avoid multicollanarity
num <- c("tenure","MonthlyCharges")
allvar <- c(fac,num)

#Checking for outliars for numeric data 
ggplot(data = customer, mapping = aes(y = MonthlyCharges)) +
  geom_boxplot()

ggplot(data = customer, mapping = aes(y = tenure)) +
  geom_boxplot()
#There are no outliars in our numeric data.


#CART
#Recurive partionating 
#only allows  Binary split 
#for grouped: grouping needs to preserve the order for ordinal var
#for contineus numerical variables we need to bin it

##DECISION TREE##

set.seed(230)
# Create train and test subsets
sub <- createDataPartition(y = customer$Churn, # target variable
                           p = 0.80, # % in training
                           list = FALSE)

train <- customer[sub, ] 
test <- customer[-sub, ] 

#ANALYSIS
customer.rpart <- rpart(formula = Churn ~ ., # Y ~ all other variables in dataframe
                        data = train[  ,c(allvar,"Churn")],
                        cp=0.05,
                        method = "class")

# Decision Tree model
#gives us the split information
customer.rpart

# We can use the variable.importance
# component of the rpart object to 
# obtain variable importance
#HOw imp each of the var are in building the tree
customer.rpart$variable.importance

#contract, totalCharges are most imp factors.

## Tree Plots
par(mar = c(2, 2, 2, 4))

prp(x = customer.rpart,
    type=5,
    branch= 0.3,
    under=TRUE,
    extra = 2) # include proportion of correct predictions

rpart.plot(customer.rpart, box.palette="RdBu", shadow.col="gray", nn=TRUE)


#rpart.plot(customer.rpart,
#           type=5,
 #          branch= 0.3,
  #         under=TRUE)

## Training Performance
# We use the predict() function to generate 
# class predictions for our training set
base.trpreds <- predict(object = customer.rpart, # DT model
                        newdata = train, # training data
                        type = "class")
#confusion matrix
DT_train_conf <- confusionMatrix(data = base.trpreds, # predictions
                                 reference = train$Churn, # actual
                                 positive = "Yes",
                                 mode = "everything")
DT_train_conf



## Testing Performance
# We use the predict() function to generate 
# class predictions for our testing set
base.tepreds <- predict(object = customer.rpart, # DT model
                        newdata = test, # training data
                        type = "class")

# We can use the confusionMatrix() function
# from the caret package to obtain a 
# confusion matrix and obtain performance
# measures for our model applied to the
# testing dataset (test).
DT_test_conf <- confusionMatrix(data = base.tepreds, # predictions
                                reference = test$Churn, # actual
                                positive = "Yes",
                                mode = "everything")
DT_test_conf

## Goodness of Fit

# To assess if the model is balanced,
# underfitting or overfitting, we compare
# the performance on the training and
# testing. We can use the cbind() function
# to compare side-by-side.

# Overall
cbind(Training = DT_train_conf$overall,
      Testing = DT_test_conf$overall)

# Class-Level
cbind(Training = DT_train_conf$byClass,
      Testing = DT_test_conf$byClass)

### Hyperparameter Tuning Model ##
  
grids <- expand.grid(cp = seq(from = 0,
                                to = 0.05,
                                by = 0.005))
grids
ctrl <- trainControl(method = "repeatedcv",
                     number = 10,
                     repeats = 3,
                     search = "grid")
set.seed(555)
DTFit <- train(form = Churn ~ ., # use all variables in data to predict churn
               data = train[  ,c(allvar,"Churn")], # include only relevant variables
               method = "rpart", # use the rpart package
               trControl = ctrl, # control object
               tuneGrid = grids) 
DTFit

# We can plot the cp value vs. Accuracy
plot(DTFit)
confusionMatrix(DTFit)
varImp(DTFit)
#the graph shows that highest accuracy is at 0.05 cp. 

#Tuned model performance
tune.trpreds <- predict(object = DTFit,
                        newdata = train)

DT_trtune_conf <- confusionMatrix(data = tune.trpreds, # predictions
                                  reference = train$Churn, # actual
                                  positive = "Yes",
                                  mode = "everything")
DT_trtune_conf

## Testing Performance ##

tune.tepreds <- predict(object = DTFit,
                        newdata = test)

DT_tetune_conf <- confusionMatrix(data = tune.tepreds, # predictions
                                  reference = test$Churn, # actual
                                  positive = "Yes",
                                  mode = "everything")
DT_tetune_conf

# Overall
cbind(Training = DT_trtune_conf$overall,
      Testing = DT_tetune_conf$overall)

# Class-Level
cbind(Training = DT_trtune_conf$byClass,
      Testing = DT_tetune_conf$byClass)

#----------------#
##RANDOM FOREST##
ctrl <- trainControl(method = "repeatedcv",
                     number = 10,
                     repeats = 3,
                     search = "grid")


floor(sqrt(length(allvar)))
grids <- expand.grid(mtry = seq(from = 2,
                                to = 9,
                                by = 1))
set.seed(212)
RFFit <- train(x = train[ ,allvar],
               y = train$Churn,
               method = "rf",
               trControl = ctrl,
               tuneGrid = grids)

RFFit
##Accuracy was used to select the optimal model using the
##largest value.
##The final value used for the model was mtry = 2
plot(varImp(RFFit))

confusionMatrix(RFFit)

tune.trpreds <- predict(object = RFFit,
                        newdata = train)

RF_tRFune_conf <- confusionMatrix(data = tune.trpreds, # predictions
                                  reference = train$Churn, # actual
                                  positive = "Yes",
                                  mode = "everything")
RF_tRFune_conf

tune.tepreds <- predict(object = RFFit,
                        newdata = test)

RF_tetune_conf <- confusionMatrix(data = tune.tepreds, # predictions
                                  reference = test$Churn, # actual
                                  positive = "Yes",
                                  mode = "everything")
RF_tetune_conf

# Overall
cbind(Training = RF_tRFune_conf$overall,
      Testing = RF_tetune_conf$overall)

# Class-Level
cbind(Training = RF_tRFune_conf$byClass,
      Testing = RF_tetune_conf$byClass)

save.image(file="FinalProject_Group4.RData")

