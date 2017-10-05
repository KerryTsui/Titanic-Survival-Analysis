from DataCleansing import *
from model import *
from correlation import *

test_data['Embarked'] = test_data['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)
test_data['Sex'] = test_data['Sex'].map( {'male': 1, 'female': 0} ).astype(int)
test_features = ['Sex', 'Age', 'Pclass', 'Parch', 'SibSp']
test_x = test_data.loc[:, test_features]
pred_y = logreg.predict(test_x)