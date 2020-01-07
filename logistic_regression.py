from sklearn.linear_model import LogisticRegression
import constants
import util

# %%
[x_train, y_train] = util.read(constants.TRAIN)

linear = LogisticRegression()
linear.fit(x_train, y_train)

util.evalute(linear, x_train, y_train)

# %%
[x_test, y_test] = util.read(constants.TEST)

util.evalute(linear, x_test, y_test)

# %%
[x_prove, y_prove] = util.read(constants.PROVE)

util.evalute(linear, x_prove, y_prove)
