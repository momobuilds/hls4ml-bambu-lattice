import tensorflow as tf
from tensorflow.keras import layers, models

# 1. Load MNIST Data
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0  # Normalize to [0, 1]
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

# 2. Define the "Tiny-CNN"
model = models.Sequential([
    layers.Input(shape=(28, 28, 1)),
    layers.Conv2D(4, (3, 3), activation='relu', name='conv1'),
    layers.MaxPooling2D((2, 2), name='pool1'),
    layers.Flatten(),
    layers.Dense(10, activation='softmax', name='output')
])

# 3. Compile and Train
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

print("Starting training...")
model.fit(x_train, y_train, epochs=3, validation_split=0.1, batch_size=32)

# 4. Save the model
model.save('models/tiny_mnist_model.h5')
print("Model saved to models/tiny_mnist_model.h5")