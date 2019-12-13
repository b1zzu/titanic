#!/usr/bin/env python

# Try to train a linear regression model

from sklearn.linear_model import LinearRegression
import pandas
import constants


def prepare(data: pandas.DataFrame) -> [pandas.DataFrame, pandas.DataFrame]:
    x = data[['Pclass', 'Age']].copy()
    y = data['Survived'].copy()

    # Fill NaN with the mean of all others
    x['Age'].fillna(round(x['Age'].mean(), 1), inplace=True)

    # Power of Age
    x['Age^2'] = x['Age'].pow(2)

    return [x, y]


train = pandas.read_csv(constants.TRAIN)
test = pandas.read_csv(constants.TEST)

[x_train, y_train] = prepare(train)
[x_test, y_test] = prepare(test)

print("\nTrain:")
print(x_train.head())

print("\nTest:")
print(x_test.head())

linear = LinearRegression(normalize=True)
linear.fit(x_train, y_train)

print("\nScore:")
print("Train: {}".format(linear.score(x_train, y_train)))

print("Test: {}".format(linear.score(x_test, y_test)))
