from DataCleansing import *
import sklearn
from sklearn.linear_model import LogisticRegression

features = ['Sex', 'Age', 'Pclass', 'Parch', 'SibSp']
train_x = train_data.loc[:, features]
train_y = train_data['Survived']
logreg = LogisticRegression().fit(train_x, train_y)