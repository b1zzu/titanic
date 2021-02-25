
from tensorflow import keras
import constants
import util

[x_train, y_train] = util.readd(constants.TRAIN)
[x_test, y_test] = util.readd(constants.TEST)
[x_prove, y_prove] = util.readd(constants.PROVE)

model = keras.models.Sequential()
model.add(keras.layers.Dense(6, input_dim=6, activation='relu'))
model.add(keras.layers.Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5000, batch_size=512)
model.save('deep_model_v2.h5')


r = model.evaluate(x_train, y_train)
print(r)

r = model.evaluate(x_test, y_test)
print(r)

r = model.evaluate(x_prove, y_prove)
print(r)
