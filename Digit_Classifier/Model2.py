import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist

# 1) Load data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 2) Preprocess
x_train = x_train[..., None] / 255.0  # (N,28,28,1)
x_test  = x_test[..., None] / 255.0

# 3) Build a tiny CNN
model = models.Sequential([
    layers.Conv2D(32, 3, activation='relu', input_shape=(28,28,1)),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 4) Train
model.fit(x_train, y_train, epochs=5, batch_size=128, validation_split=0.1, verbose=2)

# 5) Evaluate
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f"Test accuracy: {test_acc:.4f}")
