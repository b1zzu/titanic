
# Try to train a linear regression model

# %%
from sklearn.ensemble import RandomForestClassifier
import pandas
import constants
import matplotlib.pyplot as plt


def prepare(data: pandas.DataFrame, age_fill: int) -> [pandas.DataFrame, pandas.Series]:
    x = data[['Age', 'Pclass', 'SibSp', 'Parch', 'Fare']].copy()
    y = data['Survived'].copy()

    # Fill Sex, Embarked
    x['Sex'] = pandas.factorize(data['Sex'])[0]

    # Doesn't improve
    # x['Embarked'] = pandas.factorize(data['Embarked'])[0]

    # Fill NaN with the mean of all others
    x['Age'].fillna(age_fill, inplace=True)

    return [x, y]


def evalute(model: RandomForestClassifier, x: pandas.DataFrame, y: pandas.DataFrame):

    score = model.score(x, y)

    prediction = linear.predict(x)
    prediction[prediction >= 0.5] = 1
    prediction[prediction < 0.5] = 0
    result = prediction == y
    per = result.value_counts(normalize=True)[True]

    return [score, per]


# %%
train = pandas.read_csv(constants.TRAIN)
age_fill = round(train['Age'].median(), 1)
[x_train, y_train] = prepare(train, age_fill)

linear = RandomForestClassifier(min_samples_split=6)
linear.fit(x_train, y_train)

evalute(linear, x_train, y_train)

# %%
test = pandas.read_csv(constants.TEST)
[x_test, y_test] = prepare(test, age_fill)
evalute(linear, x_test, y_test)

# %%
prove = pandas.read_csv(constants.PROVE)
[x_prove, y_prove] = prepare(prove, age_fill)
evalute(linear, x_prove, y_prove)



# %%
