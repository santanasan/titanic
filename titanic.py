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
train_data['Ticket_class'] = train_data['Ticket'].str.extract(r'([A-Za-z]+)')
train_data['Ticket_class'] = train_data['Ticket_class'].fillna('absent')
train_data['Ticket_number'] = train_data['Ticket'].str.extract(r'([0-9]{3,})')
train_data['Ticket_number'] = train_data['Ticket_number'].fillna(0)
train_data['Age'] = train_data['Age'].fillna('999')
train_data['Embarked'] = train_data['Embarked'].fillna('999')
train_data['Fare'] = train_data['Fare'].fillna('0')
train_data['Relatives'] = train_data['SibSp'] + train_data['Parch']

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
    
#train_data['Survived'].corr(train_data['Name'])    
#    y = train_data['Survived'].corr(train_data[i])
#    x.append(y)
#x = [train_data['Survived'].corr(train_data[i]) for i in list(train_data.columns)]
#train_data['Survived'].corr(train_data['Pclass'])
#list(train_data.columns)
#range(0,len(train_data.columns))

# %% [code]
# import test data

test_data = pd.read_csv(r'C:\Users\thiag\.kaggle\titanic\test.csv')
test_data['Ticket_class'] = test_data['Ticket'].str.extract(r'([A-Za-z]+)')
test_data['Ticket_class'] = test_data['Ticket_class'].fillna('absent')
test_data['Ticket_number'] = test_data['Ticket'].str.extract(r'([0-9]{3,})')
test_data['Ticket_number'] = test_data['Ticket_number'].fillna(0)
test_data['Age'] = test_data['Age'].fillna('999')
test_data['Embarked'] = test_data['Embarked'].fillna('999')
test_data['Fare'] = test_data['Fare'].fillna('0')
test_data['Relatives'] = test_data['SibSp'] + test_data['Parch']

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
# from sklearn.ensemble import RandomForestClassifier

# y = train_data["Survived"]

# features = ["Pclass", "Sex", "SibSp", "Parch"]
# X = pd.get_dummies(train_data[features])
# X_test = pd.get_dummies(test_data[features])

# model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
# model.fit(X, y)
# predictions = model.predict(X_test)

# %% [code]

def transform(df):
    df['Sex'] = np.where((df.Sex == 'male'), 1 ,df.Sex)
    df['Sex'] = np.where((df.Sex == 'female'),0 ,df.Sex)
    df.Sex = pd.to_numeric(df.Sex)
    df['Embarked'] = np.where((df.Embarked == 'C'), 0 ,df.Embarked)
    df['Embarked'] = np.where((df.Embarked == 'Q'), 1 ,df.Embarked)
    df['Embarked'] = np.where((df.Embarked == 'S'), 2 ,df.Embarked)
    df.Embarked = pd.to_numeric(df.Embarked)
    wxy = df['Ticket_class'].unique()
    for i in range(0, len(wxy)):
        df['Ticket_class'] = np.where((df.Ticket_class == wxy[i]),
                                      i, df.Ticket_class)
    return df.convert_dtypes()

X_train = train_data.copy()
X_train = X_train.drop(['PassengerId', 'Name', 'Ticket', 'Cabin','Survived'], axis=1)
X_train = transform(X_train).astype(float)
X_test = test_data.copy()
X_test = X_test.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
X_test = transform(X_test).astype(float)

y = train_data['Survived']

from sklearn.model_selection import GridSearchCV

import lightgbm as lgb

# other scikit-learn modules
estimator = lgb.LGBMClassifier()

param_grid = {
    'boosting': ['dart', 'gbdt'],
    'learning_rate': [0.01, 0.1, 0.15],
    'num_iterations': [150, 200, 300],
    'n_estimators': [40, 60, 80],
    'num_leaves': [30, 45, 60],
#    'num_threads': [4]
    # 'application': ['binary'],
     'objective': ['binary'],
     'metric': ['binary_logloss']
    # 'feature_fraction':[ 0.5],
    # 'bagging_fraction': [0.5],
    # 'bagging_freq': [20],
    # 'verbose': [0]
}

gs = GridSearchCV(estimator, param_grid, cv=5, n_jobs=-1)
gs_fit = gs.fit(X_train, y)

dfresults = pd.DataFrame(gs_fit.cv_results_).sort_values('mean_test_score', ascending=False)[0:10]
predictions = gs.predict(X_test)

print('Best parameters found by grid search are:', gbm.best_params_)
# %% [code]

from sklearn.model_selection import GridSearchCV

from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier()
param_grid = {'n_estimators': [10, 150, 300],
        'max_depth': [30, 60, 90, None]}

gs = GridSearchCV(rf, param_grid, cv=5, n_jobs=-1)
gs_fit = gs.fit(X_train, y)

dfresults =  pd.DataFrame(gs_fit.cv_results_).sort_values('mean_test_score', ascending=False)[0:10]
predictions = gs.predict(X_test)

print('Best parameters found by grid search are:', gs.best_params_)
# %%[code]
from sklearn.model_selection import GridSearchCV

from sklearn.ensemble import GradientBoostingClassifier

param_grid = {
    'loss': ['deviance', 'exponential'],
    'learning_rate': [0.1, 0.15, 0.2],
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 4, 5],
    'random_state': [0],
    'max_features': ['auto', 'sqrt', 'log2']
}

clf = GradientBoostingClassifier(
    n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)



gs = GridSearchCV(clf, param_grid, cv=5, n_jobs=-1)
gs_fit = gs.fit(X_train, y)

dfresults =  pd.DataFrame(gs_fit.cv_results_).sort_values('mean_test_score', ascending=False)[0:10]

predictions = gs.predict(X_test)

output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
output.to_csv('my_submission.csv', index=False)
print("Your submission was successfully saved!")