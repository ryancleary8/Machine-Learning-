from tensorflow import keras
import numpy as np

X = np.array([[1], [3]], dtype=float)
y = np.array([[3], [5]], dtype=float)

model = keras.Sequential([
    keras.layers.Dense(units=2, input_shape=[1]),
    keras.layers.Dense(2)
])

# Compile it: use mean squared error for loss
model.compile(optimizer='sgd', loss='mean_squared_error')

# Train it
model.fit(X, y, epochs=500, verbose=0)

weights, bias = model.layers[0].get_weights()
print("Weights:", weights)
print("Bias:", bias)

# Test it
pred = model.predict(np.array([[5]], dtype=float), verbose=0)
print(float(pred[0, 0]))  # Expected ~11 