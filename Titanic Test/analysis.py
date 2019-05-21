import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

# Load training and testing data.
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# Taking a look at the data.
print(train.head())
train.info() ## Missing Embarked, Age, Cabin
print(train.describe())
print(train.Survived.value_counts(normalize=True))

# Grabbing features.
feats = train.columns.values
print(feats) ## 'Survived' 'Pclass' 'Name' 'Sex' 'Age' 'SibSp' 'Parch' 'Ticket' 'Fare' 'Cabin' 'Embarked'

fig, axes = plt.subplots(2, 4, figsize=(16, 10))
sns.countplot('Survived', data=train, ax=axes[0, 0])
sns.countplot('Pclass', data=train, ax=axes[0,1])
sns.countplot('Sex', data=train, ax=axes[0,2])
sns.countplot('SibSp', data=train, ax=axes[0,3])
sns.countplot('Parch', data=train, ax=axes[1,0])
sns.countplot('Embarked' ,data=train, ax=axes[1,1])
sns.distplot(train['Fare'], kde=True, ax=axes[1,2])
sns.distplot(train['Age'].dropna(), kde=True, ax=axes[1,3])

print(train.corr()) ## Survived&Pclass, Survived&Fare, Pclass&Age, Pclass&Fare, Age&SibSp, SibSp&Parch

# Filling in Missing Values
print(train['Embarked'].describe()) ## S is most common
train['Embarked'] = train['Embarked'].fillna('S')
train.Age.fillna(train.Age.mean(), inplace=True)

# Feature Engineering
train['Family'] = train.SibSp + train.Parch
train['Title'] = train.Name.str.extract('\, ([A-Z][^ ]*\.)',expand=False)
train.Title.fillna('Mr.', inplace=True)
print(train.Title.value_counts().reset_index())
train['Cabin_Num'] = train.Cabin.str[0]

print(train.isnull().sum())

feats = train.columns.values
train_feats = ['Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Embarked', 'Family', 'Title']

train_data = train[train_feats]

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

lr = LogisticRegression()
X = train_data[['Age']].values
y = train_data['Survived'].values

lr.fit(X,y)

y_predict = lr.predict(X)
print((y == y_predict).mean())

rf = RandomForestClassifier()
X = train_data[['Age', 'SibSp', 'Parch', 'Fare', 'Family']]
Y = train_data['Survived']

rf.fit(X, Y)

print((y == rf.predict(X)).mean())

# Creating Test Predictions
print(test.describe())
test.Age.fillna(test.Age.mean(), inplace=True)
test['Embarked'] = test['Embarked'].fillna('S')
test['Fare'] = test['Fare'].fillna('0')
print(test.describe())
test['Family'] = test.SibSp + test.Parch
result = rf.predict(test[['Age', 'SibSp', 'Parch', 'Fare', 'Family']])
submit = pd.DataFrame({'PassengerId':test['PassengerId'],'Survived':result})
submit.to_csv('trial1.csv', index=False)
