import tensorflow as tf
# Trouble shooting for SSL: CERTIFICATE_VERIFY_FAILED issue when loading data
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
from tensorflow.keras.layers import Flatten, Dense
import numpy as np

mnist = tf.keras.datasets.fashion_mnist

(training_images, training_labels), (test_images, test_labels) = mnist.load_data()
training_images = training_images / 255.0
test_images = test_images / 255.0


class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs.get('loss') < 0.4:
            print("Reached 60% accuracy thus stopping the training")
            self.model.stop_training = True


model = tf.keras.models.Sequential([
    Flatten(input_shape=(28, 28)),  # turns the matrix into a 1-Dimensional set
    Dense(128, activation=tf.nn.relu),
    Dense(10, activation=tf.nn.softmax)
])
my_callback = myCallback()
model.compile(
    optimizer=tf.optimizers.Adam(),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
# Optionally set callbacks
model.fit(training_images, training_labels, epochs=5, callbacks=[myCallback])

model.evaluate(test_images, test_labels)

classification = model.predict(test_images)

# let's check a few result
for i in range(3):
    y_hat = np.argmax(classification[i])
    y = test_labels[i]
    print(f'predicted: {y_hat}, answer: {y}')
