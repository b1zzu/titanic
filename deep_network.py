
from tensorflow import keras
import constants
import util
import numpy

[x_train, y_train] = util.read(constants.TRAIN)
model = keras.models.Sequential()
model.add(keras.layers.Dense(32, input_dim=6, activation='relu'))
model.add(keras.layers.Dense(1, activation='sigmoid'))
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
# labels = keras.utils.to_categorical(y_train, num_classes=10)

model.fit(x_train.values, numpy.asarray(y_train), epochs=5000, batch_size=512)
model.save('deep_model_v2.h5')


r = model.evaluate(x_train.values, numpy.asarray(y_train))
print(r)

[x_test, y_test] = util.read(constants.TEST)
r = model.evaluate(x_test.values, numpy.asarray(y_test))
print(r)

[x_prove, y_prove] = util.read(constants.PROVE)
r = model.evaluate(x_prove.values, numpy.asarray(y_prove))
print(r)
