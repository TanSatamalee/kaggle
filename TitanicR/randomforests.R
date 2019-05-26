# Visualization
library(ggplot2)
library(ggthemes)
library(scales)
# Data Manipulation
library(dplyr)
# Imputation
library(mice)
# Classification Algorithm
library(randomForest)


# Loads training and test datasets
trainSet <- read.csv("train.csv", stringsAsFactors=FALSE)
testSet <- read.csv("test.csv", stringsAsFactors=FALSE)
fullSet <- bind_rows(trainSet, testSet)


# Extract title from passenger's name
fullSet$Title <- gsub('(.*, )|(\\..*)', '', fullSet$Name)
# Titles with the smallest frequency for combining
rare_title <- c('Capt', 'Col', 'Don', 'Dona', 'Dr', 'Jonkheer', 'Lady', 'Major', 'Rev', 'Sir', 'the Countess')
fullSet$Title[fullSet$Title == 'Mlle'] <- 'Miss'
fullSet$Title[fullSet$Title == 'Ms'] <- 'Miss'
fullSet$Title[fullSet$Title == 'Mme'] <- 'Mrs'
fullSet$Title[fullSet$Title %in% rare_title] <- 'Rare'
# Extract surname from passenger's name
fullSet$Surname <- sapply(fullSet$Name, FUN=function(x) {strsplit(x, split='[,.]')[[1]][1]})

# Counting the family size of the passenger
fullSet$FamilySize <- fullSet$SibSp + fullSet$Parch + 1
fullSet$Family <- paste(fullSet$Surname, fullSet$FamilySize, sep='_')

# ggplot2 relationship between familysize and survival
ggplot(fullSet[1:891,], aes(x = FamilySize, fill = factor(Survived))) +
  geom_bar(stat='count', position='dodge') +
  scale_x_continuous(breaks=c(1:11)) +
  labs(x = 'Family Size') +
  theme_few()

# Separating between family size
fullSet$FSD[fullSet$FamilySize == 1] <- 'single'
fullSet$FSD[fullSet$FamilySize < 5 & fullSet$FamilySize > 1] <- 'small'
fullSet$FSD[fullSet$FamilySize > 4] <- 'large'

# Mosaic plot of family size (FSD) and survival
mosaicplot(table(fullSet$FSD, fullSet$Survived), main='Family Size by Survival', shade=TRUE)

# Checks to see if passenger was alone.
fullSet$Alone <- 0
fullSet$Alone[fullSet$FamilySize == 1] <- 1

# Extracting the deck from cabin
fullSet$Deck <- factor(sapply(fullSet$Cabin, function(x) strsplit(x, NULL)[[1]][1]))

# ggplot2 for embarkment, passenger class, and fare
embark_fare <- fullSet %>% filter(PassengerId != 62 & PassengerId != 830)
ggplot(embark_fare, aes(x = Embarked, y = Fare, fill = factor(Pclass))) +
  geom_boxplot() +
  geom_hline(aes(yintercept=80), colour='red', linetype='dashed', lwd=2) +
  scale_y_continuous(labels=dollar_format()) +
  theme_few()

# Replace the missing values for poeple with fare of $80
fullSet$Embarked[c(62, 830)] <- 'C'

# ggplot2 for passengers sharing same class and embark as passenger 1044
ggplot(fullSet[fullSet$Pclass == '3' & fullSet$Embarked == 'S', ], 
       aes(x = Fare)) +
  geom_density(fill = '#99d6ff', alpha=0.4) + 
  geom_vline(aes(xintercept=median(Fare, na.rm=T)),
             colour='red', linetype='dashed', lwd=1) +
  scale_x_continuous(labels=dollar_format()) +
  theme_few()

# Replacing the Fare for passenger 1044 with the median of common passengers
fullSet$Fare[1044] <- median(fullSet[fullSet$Pclass == '3' & fullSet$Embarked == 'S', ]$Fare, na.rm = TRUE)

# Creating model for predicting ages based on other variables
factor_vars <- c('PassengerId', 'Pclass', 'Sex', 'Embarked', 'Title', 'Surname', 'Family', 'FSD')
fullSet[factor_vars] <- lapply(fullSet[factor_vars], function(x) as.factor(x))
mice_mod <- mice(fullSet[, !names(fullSet) %in% c('PassengerId','Name','Ticket','Cabin','Family','Surname','Survived')], method='rf') 
mice_output <- complete(mice_mod)

# Comparing mice and original data
par(mfrow=c(1, 2))
hist(fullSet$Age, freq=F, main='Age:Original Data', col='darkgreen', ylim=c(0, 0.04))
hist(mice_output$Age, freq=F, main='Age:Mice Output', col='lightgreen', ylim=c(0, 0.04))

# Replace the original data with mice
fullSet$Age <- mice_output$Age

# ggplot2 for relationship between age and survival
par(mfrow=c(1, 1))
ggplot(fullSet[1:891,], aes(Age, fill = factor(Survived))) + 
  geom_histogram() + 
  facet_grid(.~Sex) + 
  theme_few()

# Create a child category
fullSet$Child[fullSet$Age < 9] <- 'Child'
fullSet$Child[fullSet$Age >= 9 & fullSet$Age < 18] = 'Teen'
fullSet$Child[fullSet$Age >= 18] <- 'Adult'

# Create a mother category
fullSet$Mother <- 'Not Mother'
fullSet$Mother[fullSet$Sex == 'female' & fullSet$Parch > 0 & fullSet$Age >= 18 & fullSet$Title == 'Mrs'] <- 'Mother'
fullSet$Child  <- factor(fullSet$Child)
fullSet$Mother <- factor(fullSet$Mother)

# Check all variables
md.pattern(fullSet)


# Split back to train and test
train <- fullSet[1:891, ]
test <- fullSet[892:1309, ]

# Build randomforest model
set.seed(214)
model <- randomForest(factor(Survived) ~ Pclass + Sex + FamilySize + 
                        Fare + Title + FSD + Child + Mother, data=train,
                        ntree=300)
print(model)

# Plotting the model error
plot(model, ylim=c(0,0.36))
legend('topright', colnames(model$err.rate), col=1:3, fill=1:3)

# Plotting importance of variables
importance <- importance(model)
varImportance <- data.frame(Variables = row.names(importance), Importance = round(importance[ ,'MeanDecreaseGini'],2))
rankImportance <- varImportance %>% mutate(Rank = paste0('#', dense_rank(desc(Importance))))
ggplot(rankImportance, aes(x = reorder(Variables, Importance), 
                           y = Importance, fill = Importance)) +
  geom_bar(stat='identity') + 
  geom_text(aes(x = Variables, y = 0.5, label = Rank),
            hjust=0, vjust=0.55, size = 4, colour = 'red') +
  labs(x = 'Variables') +
  coord_flip() + 
  theme_few()


# Predicting
prediction <- predict(model, test)
solution <- data.frame(PassengerID = test$PassengerId, Survived = prediction)
write.csv(solution, file='randomforests_sub1.csv', row.names=FALSE)
