library(dplyr)
library(ggplot2)
library(rpart)
library(randomForest)

train <- read.csv("data/train.csv", stringsAsFactors=FALSE)
test <- read.csv("data/test.csv", stringsAsFactors=FALSE)
full <- bind_rows(train, test)

## Figuring out how many missing values we have
sapply(full, function(x) sum(is.na(x) | x == ''))
# Missing:
#   Item_Outlet_Sales: 5681
#   Outlet_Size: 4016
#   Item_Weight: 2439

## Figuing out how many items in store
sapply(full, function(x) length(unique(x)))
# From Item_Identifier, 1559 unique items

## Figuring out values of each category
sapply(full, function(x) table(x))
# Outlet_Type: Grocery, Supermarket Type[1,2,3]
# Outlet_Location_Type: Tier [1,2,3]
# Outlet_Size: High, Medium, Small
# Outlet_Identifier: OUT***
# Item_Type: Baking Goods, Breads, etc. (16 total)
# Item_Fat_Content: Low Fat, Regular (needs to be combined)


## Merging similar variables in Item_Fat_Content
full$Item_Fat_Content[full$Item_Fat_Content == 'low fat' | full$Item_Fat_Content == 'LF'] <- 'Low Fat'
full$Item_Fat_Content[full$Item_Fat_Content == 'reg'] <- 'Regular'

## Replacing missing Item_Weight with average weight of same item
avg_weight <- aggregate(full$Item_Weight, list(full$Item_Identifier), FUN=mean, na.rm=TRUE, na.action=NULL)
full$Item_Weight <- ifelse(is.na(full$Item_Weight), avg_weight$x[avg_weight$Group.1 %in% full$Item_Identifier], full$Item_Weight)

## Replacing missing Outlet_Size with mode according to Outlet_Type
Mode <- function(x) {
  ux <- unique(x)
  ux[which.max(tabulate(match(x, ux)))]
}
mode_size <- aggregate(full$Outlet_Size[full$Outlet_Size != ''], list(full$Outlet_Type[full$Outlet_Size != '']), FUN=Mode)
# The mode for the two missing types are both small so we will replace
full$Outlet_Size[full$Outlet_Size == ''] <- 'Small'

## Item visibility of 0 impossible so will replace with average visibility of same item
avg_vis <- aggregate(full$Item_Visibility, list(full$Item_Identifier), FUN=mean, na.rm=TRUE, na.action=NULL)
full$Item_Visibility <- ifelse(full$Item_Visibility == 0, avg_vis$x[avg_vis$Group.1 %in% full$Item_Identifier], full$Item_Visibility)

## Create a mean visibility for a particular item across all stores
full$Mean_Visibility <- apply(full, 1, function(z) as.numeric(z[4]) / avg_vis$x[avg_vis$Group.1 == z[1]])

## Create a general category for item type
full$General_Type <- factor(substr(full$Item_Identifier, 1, 2))

## Determining number of years outlet is opened
full$Open_Years <- 2013 - full$Outlet_Establishment_Year

## From the Item_Fat_Content need to separate the non-consumables (NC)
full$Item_Fat_Content[full$General_Type == 'NC'] <- 'NC'


## Convert combined data back to train and test sets
train <- full[1:8523, ]
test <- full[8524:14204, ]

## Baseline model (mean of sales)
pred_baseline<- mean(train$Item_Outlet_Sales)
sol_baseline <- data.frame(Item_Identifier = test$Item_Identifier, Outlet_Identifier = test$Outlet_Identifier, Item_Outlet_Sales = pred_baseline)
write.csv(sol_baseline, file='baseline.csv', row.names=FALSE)
# Gives Score of 1774

## Using Linear Regression Model
lr_model <- lm(Item_Outlet_Sales ~ Item_Fat_Content + Item_Visibility + Item_Weight + Item_Type + Item_MRP + Outlet_Identifier + 
                 Outlet_Size + Outlet_Location_Type + Outlet_Type + Mean_Visibility + General_Type + Open_Years, data=train)
pred_lr <- predict(lr_model, test)
sol_lr <- data.frame(Item_Identifier = test$Item_Identifier, Outlet_Identifier = test$Outlet_Identifier, Item_Outlet_Sales = pred_lr)
write.csv(sol_lr, file='linear_regress.csv', row.names=FALSE)
# Gives score of 1204

## Using Decision Trees
dt_model <- rpart(Item_Outlet_Sales ~ Item_Fat_Content + Item_Visibility + Item_Weight + Item_Type + Item_MRP + Outlet_Identifier + 
                   Outlet_Size + Outlet_Location_Type + Outlet_Type + Mean_Visibility + General_Type + Open_Years, data=train)
pred_dt <- predict(dt_model, test)
sol_dt <- data.frame(Item_Identifier = test$Item_Identifier, Outlet_Identifier = test$Outlet_Identifier, Item_Outlet_Sales = pred_dt)
write.csv(sol_dt, file='decision_tree.csv', row.names=FALSE)
# Gives score of 1174

## Using Random Forests
var_fact = c('Item_Fat_Content', 'Item_Type', 'Outlet_Identifier', 'Outlet_Size', 'Outlet_Location_Type', 'Outlet_Type', 'General_Type')
train[var_fact] <- lapply(train[var_fact], factor)
test[var_fact] <- lapply(test[var_fact], factor)
rf_model <- randomForest(Item_Outlet_Sales ~ Item_Fat_Content + Item_Visibility + Item_Weight + Item_Type + Item_MRP + Outlet_Identifier + 
                           Outlet_Size + Outlet_Location_Type + Outlet_Type + Mean_Visibility + General_Type + Open_Years, data=train, 
                         ntree=400, mtry=6, do.trace=TRUE)
pred_rf <- predict(rf_model, test)
sol_rf <- data.frame(Item_Identifier = test$Item_Identifier, Outlet_Identifier = test$Outlet_Identifier, Item_Outlet_Sales = pred_rf)
write.csv(sol_rf, file='randomforest.csv', row.names=FALSE)
# Gives score of 1191
