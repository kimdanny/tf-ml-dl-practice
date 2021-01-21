import tensorflow as tf
# Trouble shooting for SSL: CERTIFICATE_VERIFY_FAILED issue when loading data
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Conv2D, MaxPooling2D
import numpy as np

mnist = tf.keras.datasets.fashion_mnist

(training_images, training_labels), (test_images, test_labels) = mnist.load_data()
# (60000, 28, 28) (60000,) (10000, 28, 28) (10000,)
# print(training_images.shape, training_labels.shape, test_images.shape, test_labels.shape)

# When using CNN, your input shape need to be 3-channel
training_images = training_images.reshape(60000, 28, 28, 1)
test_images = test_images.reshape(10000, 28, 28, 1)
# Normalisation
training_images = training_images / 255.0
test_images = test_images / 255.0

model = Sequential([
    Conv2D(64, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()
model.fit(training_images, training_labels, epochs=5)
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(test_acc)



