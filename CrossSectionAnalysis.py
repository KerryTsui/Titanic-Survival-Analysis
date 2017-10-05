from DataCleansing import *
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

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
