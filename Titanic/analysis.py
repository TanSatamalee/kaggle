import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
import math
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

# Load training and testing data.
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# Take a look at training set.
feats = train.columns.values
print("Starting Features:")
print(feats)
print()

print(train.head(10))
    # PassengerId just index
    # Name seems to order Last, Title First Etc (Something) take second string for title
print()

## ===================================================================
# Order of how far ports are from start of ship.
def convert_embarked(x):
    if x == 'S':
        return 2
    if x == 'C':
        return 1
    if x == 'Q':
        return 3

def convert_sex(x):
    if x == 'male':
        return 1
    if x == 'female':
        return 0

def convert_ticket(x):
    x = x.split()[-1]
    if x.isnumeric():
        return int(x)
    return -1

def convert_cabin(x):
    x = x[0]
    if x == 'A':
        return 1
    if x == 'B':
        return 2
    if x == 'C':
        return 3
    if x == 'D':
        return 4
    if x == 'E':
        return 5
    return -1
    
def convert_title(x):
    if x == 'Capt.':
        return 1
    if x in ['Col.', 'Major.']:
        return 2
    if x in ['Rev.', 'Dr.']:
        return 3
    if x in ['Mr.', 'Sir.', 'Don.', 'Jonkheer.']:
        return 4
    if x == 'Master.':
        return 5
    if x in ['Miss.', 'Mlle.', 'Mme.', 'Ms.', 'Lady.']:
        return 6
    if x == 'Mrs.':
        return 7
    
def fill_age_naive(x):
    mu = train.Age.mean()
    sigma = train.Age.std()
    return random.randint(math.floor(mu - sigma), math.ceil(mu + sigma))

def fill_fare_naive(x):
    mu = train.Fare.mean()
    sigma = train.Fare.std()
    return random.randint(math.floor(mu - sigma), math.ceil(mu + sigma))

## ====================================================================

# Convert as much to numeric data before continuing.
train.Embarked = train.Embarked.apply(convert_embarked)
train.Sex = train.Sex.apply(convert_sex)
train.Ticket = train.Ticket.apply(convert_ticket)
train.Cabin.fillna('None', inplace=True)
train.Cabin = train.Cabin.apply(convert_cabin)
train['Title'] = train.Name.str.extract('\, ([A-Z][^ ]*\.)',expand=False)
train.Title = train.Title.apply(convert_title)

# Feature Engineering.
train['Family'] = train.SibSp + train.Parch

# Filling Missing Values.
train.Title = train.Title.fillna(5)
train.Embarked = train.Embarked.fillna(2)
### train.Age[train.Age.isnull()] = train.Age[train.Age.isnull()].apply(fill_age_naive)
train.Fare[train.Fare == 0] = train.Fare[train.Fare == 0].apply(fill_fare_naive)
train.Age = train.groupby("Title").transform(lambda x: x.fillna(x.mean())).Age

# Analysis on Age
fig, axs = plt.subplots(2, 5)
fig.set_size_inches(20, 10)
train_plot = train.dropna()
sns.boxplot('Pclass', 'Age', data=train_plot, ax=axs[0][0])
sns.boxplot('Sex', 'Age', data=train_plot, ax=axs[0][1])
sns.boxplot('SibSp', 'Age', data=train_plot, ax=axs[0][2])
sns.boxplot('Parch', 'Age', data=train_plot, ax=axs[0][3])
sns.boxplot('Ticket', 'Age', data=train_plot, ax=axs[0][4])
sns.boxplot('Fare', 'Age', data=train_plot, ax=axs[1][0])
sns.boxplot('Cabin', 'Age', data=train_plot, ax=axs[1][1])
sns.boxplot('Embarked', 'Age', data=train_plot, ax=axs[1][2])
sns.boxplot('Title', 'Age', data=train_plot, ax=axs[1][3])
sns.boxplot('Family', 'Age', data=train_plot, ax=axs[1][4])

print("Describing Data:")
print(train.describe())
print(train.info())
print(train.head())
    # Missing about 700 Cabins, 200 Age, 2 Embarked, 1 Title
    # Non-numeric: Name, Sex, Ticket, Cabin, Embarked (Fixed)
print()

print(train.corr())
    # Positive: Survived&Fare, Survived&Title, Pclass&Ticket, Pclass&Embarked, SibSp&Parch, SibSp&Fare, SibSp&Title, Parch&Fare, Parch&Title, Ticket&Embarked, Fare&Title
    # Negative: Survive&Pclass, Survived&Sex, Survived&Embarked, Pclass&Age, Pclass&Fare, Pclass&Title, Sex&SibSp, Sex&Parch, Sex&Fare, Sex&Title, Age&SibSp, Age&Parch, Age&Fare, Age&Title, Ticket&Fare, Fare&Embarked
print()

train_feats = ['Pclass', 'Sex', 'Title', 'Embarked', 'Family', 'Age', 'Fare', 'Cabin']

'''
# Tuning Hyper Parameters.\
## Folds
print('n_estimate hypertune')
kfolds = [5, 10, 20, 50, 70, 100]
for i in kfolds:
    tune_rf = RandomForestClassifier()
    print(np.mean(cross_val_score(tune_rf, X, Y, cv=i)))
    # 50 seems optimal


## n_estimators
print('n_estimate hypertune')
trees = [1, 5, 10, 20, 50, 70, 100]
for i in trees:
    tune_rf = RandomForestClassifier(n_estimators=i)
    print(np.mean(cross_val_score(tune_rf, X, Y, cv=50)))
    # 50 seems optimal

## Max Depth
print('max_depth hypertune')
max_depth = [1, 5, 10, 20, 50, 100, 200]
for i in max_depth:
    tune_rf = RandomForestClassifier(n_estimators=50, max_depth=i)
    print(np.mean(cross_val_score(tune_rf, X, Y, cv=50)))
    # 5 seems optimal

## Min Samples Split
print('min_split hypertune')
min_split = [0.00001, 0.0001, 0.001, 0.01, 0.1]
for i in min_split:
    tune_rf = RandomForestClassifier(n_estimators=50, max_depth=5, min_samples_split=i)
    print(np.mean(cross_val_score(tune_rf, X, Y, cv=50)))
    # 0.001 seems optimal

## Min Samples Leaf
print('min_split hypertune')
min_leaf = [0.00001, 0.0001, 0.001, 0.01, 0.1]
for i in min_leaf:
    tune_rf = RandomForestClassifier(n_estimators=50, max_depth=5, min_samples_split=0.001, min_samples_leaf=i)
    print(np.mean(cross_val_score(tune_rf, X, Y, cv=50)))
    # 0.0001 seems optimal

## Max Features
print('min_split hypertune')
max_feat = [1, 2, 3, 4]
for i in max_feat:
    tune_rf = RandomForestClassifier(n_estimators=50, max_depth=5, min_samples_split=0.001, min_samples_leaf=0.0001, max_features=i)
    print(np.mean(cross_val_score(tune_rf, X, Y, cv=50)))
    # 2 seems optimal
'''

rf = RandomForestClassifier(n_estimators=50, max_depth=5, min_samples_split=0.001, min_samples_leaf=0.0001, max_features=2)
X = train[train_feats]
Y = train['Survived']

print(np.mean(cross_val_score(rf, X, Y, cv=20)))


# Prepare Test Values
test.Embarked = test.Embarked.apply(convert_embarked)
test.Sex = test.Sex.apply(convert_sex)
test.Ticket = test.Ticket.apply(convert_ticket)
test.Cabin.fillna('None', inplace=True)
test.Cabin = test.Cabin.apply(convert_cabin)
test['Title'] = test.Name.str.extract('\, ([A-Z][^ ]*\.)',expand=False)
test.Title = test.Title.apply(convert_title)
test['Family'] = test.SibSp + test.Parch
test.Title = test.Title.fillna(5)
test.Embarked = test.Embarked.fillna(2)
test.Fare[test.Fare == 0] = test.Fare[test.Fare == 0].apply(fill_fare_naive)
test.Fare = test.Fare.fillna(fill_fare_naive(test))
test.Age = test.groupby("Title").transform(lambda x: x.fillna(x.mean())).Age
rf.fit(X, Y)
result = rf.predict(test[train_feats])
submit = pd.DataFrame({'PassengerId':test['PassengerId'],'Survived':result})
submit.to_csv('trial2.csv', index=False)
