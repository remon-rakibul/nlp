from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import LSTM
from matplotlib import pyplot as plt
import numpy as np

X = [[[i+j/100] for i in range(5)] for j in range(100)]
y = [(i+5)/100 for i in range(100)]

X = np.array(X, dtype=float)
y = np.array(y, dtype=float)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)

model = Sequential()

model.add(LSTM((1), batch_input_shape=(None, 5, 1), return_sequences=True))
model.add(LSTM((1), return_sequences=False))

model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['accuracy'])

model.summary()

history = model.fit(X_train, y_train, epochs=400, validation_data=(X_test, y_test))

results = model.predict(X_test)

plt.scatter(range(20), results, c='r')
plt.scatter(range(20), y_test, c='g')

plt.plot(history.history['loss'])
plt.show()

