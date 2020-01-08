# %%
from tensorflow import keras
import constants
import util
import numpy


# %%
[x_train, y_train] = util.read(constants.TRAIN)
model = keras.models.load_model('deep_model.h5')

# model.summary()

# %%
model.evaluate(x_train.values, numpy.asarray(y_train))

# %%
[x_test, y_test] = util.read(constants.TEST)
model.evaluate(x_test.values, numpy.asarray(y_test))

# %%
[x_prove, y_prove] = util.read(constants.PROVE)
model.evaluate(x_prove.values, numpy.asarray(y_prove))
