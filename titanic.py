# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

# %% [code]
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk(r'C:\Users\thiag\.kaggle\titanic'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

# %% [code]
# import train data:
train_data = pd.read_csv(r'C:\Users\thiag\.kaggle\titanic\train.csv')
train_data.head()

# %% [code]
train_data2 = train_data.copy()
train_data2.drop(['Name', 'Ticket'], axis = 1)

# %% [code]

# puttin in equation the first letter of the cabin, supposing the region of
# the passenger plays a role in their survival

x = list(train_data2['Cabin'].values)

for i in range(0, len(x)):
    if pd.isna(x[i]):
        x[i] = int(99)
    else:
        x[i] = ord(x[i][0])

train_data2['Cabin'] = x


x = list(train_data2['Embarked'].values)

for i in range(0, len(x)):
    if pd.isna(x[i]):
        x[i] = int(99)
    else:
        x[i] = ord(x[i][0])

train_data2['Embarked'] = x

# %% [code]
x

# %% [code]
#train_data['Survived'].corr(train_data[train_data.columns[1]])
x = []
for i in list(train_data.columns):
    print(i)
    
train_data['Survived'].corr(train_data['Name'])    
#    y = train_data['Survived'].corr(train_data[i])
#    x.append(y)
#x = [train_data['Survived'].corr(train_data[i]) for i in list(train_data.columns)]
#train_data['Survived'].corr(train_data['Pclass'])
#list(train_data.columns)
#range(0,len(train_data.columns))

# %% [code]
# import test data

test_data = pd.read_csv(r'C:\Users\thiag\.kaggle\titanic\test.csv')
test_data.head()

# %% [code]
women = train_data.loc[train_data.Sex == 'female']["Survived"]
rate_women = sum(women)/len(women)

print("% of women who survived:", rate_women)

# %% [code]
men = train_data.loc[train_data.Sex == 'male']["Survived"]
rate_men = sum(men)/len(men)

print("% of men who survived:", rate_men)

# %% [code]
from sklearn.ensemble import RandomForestClassifier

y = train_data["Survived"]

features = ["Pclass", "Sex", "SibSp", "Parch"]
X = pd.get_dummies(train_data[features])
X_test = pd.get_dummies(test_data[features])

model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
model.fit(X, y)
predictions = model.predict(X_test)

output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
output.to_csv('my_submission.csv', index=False)
print("Your submission was successfully saved!")