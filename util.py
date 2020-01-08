import pandas
import numpy


def read(file: str) -> [pandas.DataFrame, pandas.Series]:
    d = pandas.read_csv(file)
    return prepare(d)


def prepare(data: pandas.DataFrame) -> [pandas.DataFrame, pandas.Series]:
    x = data[['Age', 'Pclass', 'SibSp', 'Parch', 'Fare']].copy()
    y = data['Survived'].copy()

    # Fill Sex, Embarked
    x['Sex'] = pandas.factorize(data['Sex'])[0]

    # Doesn't improve
    # x['Embarked'] = pandas.factorize(data['Embarked'])[0]

    # Fill NaN with the mean of all others
    x['Age'].fillna(round(x['Age'].mean(), 1), inplace=True)

    return [x, y]


def readd(file: str):
    [x, y] = read(file)
    return [x.values, numpy.asarray(y)]
