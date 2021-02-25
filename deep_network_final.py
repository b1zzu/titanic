 # %%
from tensorflow import keras
import constants
import util
import pandas

[x, y] = util.readd(constants.ORIGINAL_TRAIN)

model = keras.models.Sequential()
model.add(keras.layers.Dense(32, input_dim=6, activation='relu'))
model.add(keras.layers.Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(x, y, epochs=50000, batch_size=512)
model.save('deep_model_final_v2.h5')

# model = keras.models.load_model('deep_model_v2.h5')

r = model.evaluate(x, y)

final = pandas.read_csv("./data/final_test.csv")
x_final = util.prepare_x(final)
y_final = model.predict(x_final)

y_final = y_final.flatten()
y_final[y_final >= 0.5] = 1
y_final[y_final < 0.5] = 0

result = final[['PassengerId']].copy()
result['Survived'] = y_final.astype(int)
result = result.set_index('PassengerId')
result.to_csv('./data/my_result_v2.csv')


# %%
