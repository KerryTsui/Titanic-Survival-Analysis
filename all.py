import csv
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from random import choice
import sklearn
import seaborn as sns
from sklearn.linear_model import LogisticRegression

train_data = pd.read_table('/Users/jiayicui/Desktop/Titanic Survival Analysis/data/train.csv', sep = ',')
print(train_data.head(5))

print(train_data.shape)
# There are 891 rows and 12 columns in the dataframe.
print(train_data.info())
# from the output we see that there's a very large portion of null value in 'Cabin'.
# The 'Age' colum has 177 null value.

# Considering that Age may be an important variable in prediction, we use the mode of age to fill in the 177 null value.
age_mode = pd.value_counts(train_data['Age']).index[0]
emb_mode = pd.value_counts(train_data['Embarked']).index[0]
train_data['Age'] = train_data['Age'].fillna(value = age_mode)
train_data['Embarked'] = train_data['Embarked'].fillna(value = 'S')

# Cleanse the test dataset
test_data = pd.read_csv('/Users/jiayicui/Desktop/Titanic Survival Analysis/data/test.csv', sep = ',')
test_age_mode = pd.value_counts(test_data['Age']).index[0]
test_emb_mode = pd.value_counts(test_data['Embarked']).index[0]
test_data['Age'] = test_data['Age'].fillna(value = test_age_mode)
test_data['Embarked'] = test_data['Embarked'].fillna(value = 'S')
test_data['Sex'] = test_data['Sex'].fillna(choice(['male', 'female']))

# Cross Section Analysis

# General survive rate
survive_rate =float(train_data['Survived'].sum())/train_data['Survived'].count()
print(survive_rate)
# the output is 38.38%

# The relationship between sex and survive rate
sex_survive=(train_data.groupby(['Sex']).sum()/train_data.groupby(['Sex']).count())['Survived']

# To look at all main correlation in one graph, since Cabin data is severely missing, we won't put it in here
g = sns.pairplot(train_data[[u'Survived', u'Pclass', u'Sex', u'Age',u'SibSp', u'Parch', u'Fare', u'Embarked']], hue='Survived', 
palette='seismic',size=1.2,diag_kind = 'kde',diag_kws=dict(shade=True),plot_kws=dict(s=10) )
g.set(xticklabels=[])
plt.show(g)

# Correlation

# To compute correlation, first convert all needed variables to numeric type
train_data['Embarked'] = train_data['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)
train_data['Sex'] = train_data['Sex'].map( {'male': 1, 'female': 0} ).astype(int)
test_data['Embarked'] = test_data['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)
test_data['Sex'] = test_data['Sex'].map( {'male': 1, 'female': 0} ).astype(int)

colormap = plt.cm.viridis
variables = train_data[[u'Survived', u'Pclass', u'Sex', u'Age', u'SibSp', u'Parch', u'Fare']]
hgr = sns.heatmap(variables.astype(float).corr(), cmap=colormap, annot=True)
plt.show(hgr)

# modeling

features = ['Sex', 'Age', 'Pclass', 'Parch', 'SibSp']
train_x = train_data.loc[:, features]
train_y = train_data['Survived']
logreg = LogisticRegression().fit(train_x, train_y)

# predicting

test_features = ['Sex', 'Age', 'Pclass', 'Parch', 'SibSp']
test_x = test_data.loc[:, test_features]
pred_y = logreg.predict(test_x)
prediction = test_data
prediction['predict_survival'] = pred_y
prediction.to_csv('/Users/jiayicui/Desktop/Titanic Survival Analysis/output/prediction.csv')