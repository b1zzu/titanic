import pandas


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


def evalute(model, x: pandas.DataFrame, y: pandas.DataFrame):

    score = model.score(x, y)

    prediction = model.predict(x)
    prediction[prediction >= 0.5] = 1
    prediction[prediction < 0.5] = 0
    result = prediction == y
    per = result.value_counts(normalize=True)[True]

    return [score, per]
