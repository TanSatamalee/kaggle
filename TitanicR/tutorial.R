library(caret)
library(randomForest)
library(fields)

# Loads training and test datasets
trainSet <- read.csv("train.csv", stringsAsFactors=FALSE)
testSet <- read.csv("test.csv", stringsAsFactors=FALSE)

# Shows the head for train and test sets
head(trainSet)
head(testSet)

# ========================================================================

# Gender Patterns
print(prop.table(table(trainSet$Survived, trainSet$Sex), 1))

# Crosstabs between Survived and Pclass
print(prop.table(table(trainSet$Survived, trainSet$Pclass), 1))

# Crosstabs between Survived and Embarked
print(prop.table(table(trainSet$Survived, trainSet$Embarked), 1))

# Crosstabs between Survived and Parch
print(prop.table(table(trainSet$Survived, trainSet$Parch), 1))
# Combine the variable with people travelled
trainSet$FamilySize <- trainSet$SibSp + trainSet$Parch + 1
testSet$FamilySize <- testSet$SibSp + testSet$Parch + 1

# Conditional box plots between Survived and Age
bplot.xy(trainSet$Survived, trainSet$Age)
print("Summary of Age Data")
print(summary(trainSet$Age))
# Creating new category for Child and Teen based on Age
trainSet$Child <- 0
trainSet$Child[trainSet$Age < 8] <- 1
print(prop.table(table(trainSet$Survived, trainSet$Child), 1))

# Conditional box plots between Survived and Fare
bplot.xy(trainSet$Survived, trainSet$Fare)
print("Summary of Fare Data")
print(summary(trainSet$Fare))
# Creating new categories for Fare
trainSet$Fare <- ifelse(is.na(trainSet$Fare), mean(trainSet$Fare, na.rm=TRUE), trainSet$Fare)
trainSet$Fare <- cut(trainSet$Fare, breaks=c(-1,10,20,30,600), labels=c('<10','10-20','20-30','30+'))

# Extract the title of each person
trainSet$Title <- sapply(trainSet$Name, FUN=function(x) {strsplit(x, split='[,.]')[[1]][2]})
trainSet$Title <- sub(' ', '', trainSet$Title)
trainSet$Title[trainSet$Title %in% c('Mme', 'Mile')] <- 'Mile'
trainSet$Title[trainSet$Title %in% c('Capt', 'Don', 'Major', 'Sir')] <- 'Sir'
trainSet$Title[trainSet$Title %in% c('Dona', 'Lady', 'the Countess', 'Jonkheer')] <- 'Lady'
trainSet$Title <- factor(trainSet$Title)

testSet$Title <- sapply(testSet$Name, FUN=function(x) {strsplit(x, split='[,.]')[[1]][2]})
testSet$Title <- sub(' ', '', testSet$Title)
testSet$Title[testSet$Title %in% c('Mme', 'Mile')] <- 'Mile'
testSet$Title[testSet$Title %in% c('Capt', 'Don', 'Major', 'Sir')] <- 'Sir'
testSet$Title[testSet$Title %in% c('Dona', 'Lady', 'the Countess', 'Jonkheer')] <- 'Lady'
testSet$Title <- factor(testSet$Title)

# Getting Surname and common Family of a Family
trainSet$Surname <- sapply(trainSet$Name, FUN=function(x) {strsplit(x, split='[,.]')[[1]][1]})
trainSet$FamilyID <- paste(as.character(trainSet$FamilySize), trainSet$Surname, sep="")
trainSet$FamilyID[trainSet$FamilySize <= 3] <- 'Small'
famIDs <- data.frame(table(trainSet$FamilyID))
famIDs <- famIDs[famIDs$Freq <= 3,]
trainSet$FamilyID[trainSet$FamilyID %in% famIDs$Var1] <- 'Small'
trainSet$FamilyID <- factor(trainSet$FamilyID)

testSet$Surname <- sapply(testSet$Name, FUN=function(x) {strsplit(x, split='[,.]')[[1]][1]})
testSet$FamilyID <- paste(as.character(testSet$FamilySize), testSet$Surname, sep="")
testSet$FamilyID[testSet$FamilySize <= 3] <- 'Small'
famIDs2 <- data.frame(table(testSet$FamilyID))
famIDs2 <- famIDs2[famIDs2$Freq <= 3,]
testSet$FamilyID[testSet$FamilyID %in% famIDs2$Var1] <- 'Small'
testSet$FamilyID <- factor(testSet$FamilyID)

# ===================================================================================

# Training Model
trainSet$Survived <- factor(trainSet$Survived)
model <- train(Survived ~ Pclass + Sex + FamilySize + Fare + Child + Title + FamilyID,
               data=trainSet, method="rf",
               trControl=trainControl(method="cv", number=5))
print(model)

# Filling NA's with mean of column
testSet$Fare <- ifelse(is.na(testSet$Fare), mean(testSet$Fare, na.rm=TRUE), testSet$Fare)
testSet$Fare <- cut(testSet$Fare, breaks=c(-1,10,20,30,600), labels=c('<10','10-20','20-30','30+'))
testSet$Child <- 0
testSet$Child[testSet$Age < 8] <- 1

# Test set predictions
testSet$Survived <- predict(model, newdata=testSet)

# Remove unneeded columns for Kaggle
submission <- testSet[,c("PassengerId", "Survived")]
write.table(submission, file="submission4.csv", col.names=TRUE, row.names=FALSE, sep=",")