import csv
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import re
import sklearn
import seaborn as sns

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
train_data['Age']=train_data['Age'].fillna(age_mode)
train_data['Embarked']=train_data['Embarked'].fillna('S')

# Cleanse the test dataset
test_data = pd.read_csv('/Users/jiayicui/Desktop/Titanic Survival Analysis/data/test.csv', sep = ',')
test_age_mode = pd.value_counts(test_data['Age']).index[0]
test_emb_mode = pd.value_counts(test_data['Embarked']).index[0]
test_data['Age']=test_data['Age'].fillna(test_age_mode)
test_data['Embarked']=test_data['Embarked'].fillna('S')