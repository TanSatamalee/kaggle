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

# Matching people with ticket numbers into the same group.
def ticket_sort(x):
    last = 0
    n = 0
    x['Group'] = 0
    x_sort = x.sort_values('Ticket')
    for i, row in x_sort.iterrows():
        if int(row['Ticket']) - last == 0:
            x_sort.loc[i, 'Group'] = n
        elif int(row['Ticket']) - last == 1:
            x_sort.loc[i, 'Group'] = n
            last = int(row['Ticket'])
        else:
            n += 1
            last = int(row['Ticket'])
            x_sort.loc[i, 'Group'] = n
    return x_sort

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
train['Family'] = train.SibSp + train.Parch + 1
train['Child'] = train.Age.apply(lambda x: 1 if x < 18 else 0)
train['Single'] = train.Family.apply(lambda x: 1 if x == 1 else 0)
train['SmallFam'] = train.Family.apply(lambda x: 1 if x < 4 else 0)
train['LargeFam'] = train.Family.apply(lambda x: 1 if x >= 4 else 0)
train = ticket_sort(train)
group_count = train.groupby("Group").count()
train['GroupSize'] = train.Group.apply(lambda x: group_count['PassengerId'][x])


# Filling Missing Values.
train.Title = train.Title.fillna(5)
train.Embarked = train.Embarked.fillna(2)
### train.Age[train.Age.isnull()] = train.Age[train.Age.isnull()].apply(fill_age_naive)
train.Fare[train.Fare == 0] = np.nan
train.Fare = train.groupby("Family").transform(lambda x: x.fillna(x.median())).Fare
train.Age = train.groupby("Title").transform(lambda x: x.fillna(x.mean())).Age

train_feats = ['Pclass', 'Sex', 'Title', 'Embarked', 'Family', 'Age', 'Fare', 'Cabin', 'SmallFam', 'GroupSize']

'''
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

# Analysis on Age
fig, axs = plt.subplots(2, 5)
fig.set_size_inches(20, 10)
train_plot = train.dropna()[train.Fare < 300]
sns.boxplot('Pclass', 'Fare', data=train_plot, ax=axs[0][0])
sns.boxplot('Sex', 'Fare', data=train_plot, ax=axs[0][1])
sns.boxplot('SibSp', 'Fare', data=train_plot, ax=axs[0][2])
sns.boxplot('Parch', 'Fare', data=train_plot, ax=axs[0][3])
sns.boxplot('Ticket', 'Fare', data=train_plot, ax=axs[0][4])
sns.boxplot('Age', 'Fare', data=train_plot, ax=axs[1][0])
sns.boxplot('Cabin', 'Fare', data=train_plot, ax=axs[1][1])
sns.boxplot('Embarked', 'Fare', data=train_plot, ax=axs[1][2])
sns.boxplot('Title', 'Fare', data=train_plot, ax=axs[1][3])
sns.boxplot('Family', 'Fare', data=train_plot, ax=axs[1][4])

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
test['Family'] = test.SibSp + test.Parch + 1
test['Child'] = test.Age.apply(lambda x: 1 if x < 18 else 0)
test['Single'] = test.Family.apply(lambda x: 1 if x == 1 else 0)
test['SmallFam'] = test.Family.apply(lambda x: 1 if x < 4 else 0)
test['LargeFam'] = test.Family.apply(lambda x: 1 if x >= 4 else 0)
test.Title = test.Title.fillna(5)
test.Embarked = test.Embarked.fillna(2)
test.Fare[test.Fare == 0] = np.nan
test.Fare = test.groupby("Family").transform(lambda x: x.fillna(x.median())).Fare
test.Age = test.groupby("Title").transform(lambda x: x.fillna(x.mean())).Age
test = ticket_sort(test)
group_count = test.groupby("Group").count()
test['GroupSize'] = test.Group.apply(lambda x: group_count['PassengerId'][x])
rf.fit(X, Y)
result = rf.predict(test[train_feats])
submit = pd.DataFrame({'PassengerId':test['PassengerId'],'Survived':result})
submit.to_csv('trial4.csv', index=False)


# Feature Importance
plt.figure()
importance = rf.feature_importances_
importance = pd.DataFrame(importance, index=X.columns, columns=["Importance"])
importance["Std"] = np.std([tree.feature_importances_ for tree in rf.estimators_], axis=0)

x = range(importance.shape[0])
y = importance.iloc[:, 0]
yerr = importance.iloc[:, 1]

plt.bar(x, y, yerr=yerr, align="center")
plt.show()
