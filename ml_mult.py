from tensorflow import keras
import numpy as np

X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]])
y = np.array([2, 12, 30, 56, 90, 132], dtype=float).reshape(-1, 1)

model = keras.Sequential([
    keras.Input(shape=(2,)),
    keras.layers.Dense(units=4, activation='relu'),
    keras.layers.Dense(units=8, activation='relu'),
    keras.layers.Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error')

model.fit(X, y, epochs=500, verbose=0)

pred = model.predict(np.array([[4, 4]], dtype=float), verbose=0)
print(pred.item())
