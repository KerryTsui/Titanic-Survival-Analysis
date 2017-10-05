from DataCleansing import *
import seaborn as sns

# To compute correlation, first convert all needed variables to numeric type
train_data['Embarked'] = train_data['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)
train_data['Sex'] = train_data['Sex'].map( {'male': 1, 'female': 0} ).astype(int)
test_data['Embarked'] = test_data['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)
test_data['Sex'] = test_data['Sex'].map( {'male': 1, 'female': 0} ).astype(int)

colormap = plt.cm.viridis
variables = train_data[[u'Survived', u'Pclass', u'Sex', u'Age', u'SibSp', u'Parch', u'Fare']]
hgr = sns.heatmap(variables.astype(float).corr(), cmap=colormap, annot=True)
plt.show(hgr)