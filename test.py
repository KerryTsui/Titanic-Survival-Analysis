from DataCleansing import *
from model import *
from correlation import *

test_features = ['Sex', 'Age', 'Pclass', 'Parch', 'SibSp']
test_x = test_data.loc[:, test_features]
pred_y = logreg.predict(test_x)
prediction = test_data['PassengerId']
prediction['pred_survive'] = pred_y
prediction.to_csv('/Users/jiayicui/Desktop/Titanic Survival Analysis/output/prediction.csv')
