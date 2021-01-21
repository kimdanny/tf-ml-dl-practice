import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

"""
Adam
RMSprop
SGD (Stochastic Gradient Descent)
"""

# Length of data are different by reviews
def sequences_shaping(sequences, dimension):
    # Make 0-filled matrix of size (len(sequences), dimension)
    results = np.zeros((len(sequences), dimension))
    for i, word_indices in enumerate(sequences):
        results[i, word_indices] = 1.0

    return results


def Visualize(histories, key='binary_crossentropy'):
    for name, history in histories:
        val = plt.plot(history.epoch, history.history['val_' + key],
                       '--', label=name.title() + ' Val')
        plt.plot(history.epoch, history.history[key], color=val[0].get_color(),
                 label=name.title() + ' Train')

    plt.xlabel('Epochs')
    plt.ylabel(key.replace('_', ' ').title())
    plt.legend()

    plt.xlim([0, max(history.epoch)])

    plt.show()


# extract upto 100 frequently-used words
word_num = 100
data_num = 25000

# IMDb dataset has 25000 train samples and 25000 test samples
(train_data, train_labels), (test_data, test_labels) = keras.datasets.imdb.load_data(num_words=word_num)

# For data shaping, we use sequence_shaping (defined above) here
train_data = sequences_shaping(train_data, dimension=word_num)
test_data = sequences_shaping(test_data, dimension=word_num)


# base_model is a simple basic model to be copied to optimizer models
base_model = keras.Sequential([
    # Need to set input_shape for the first layer
    keras.layers.Dense(128, activation=tf.nn.relu, input_shape=(word_num,)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(1, activation=tf.nn.sigmoid)
])

# optimizer models share the same base model
adam_model = base_model
rmsprop_model = base_model
sgd_model = base_model
del base_model

"""
Compile base and optimizer model
"""

adam_model.compile(
    optimizer=keras.optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999),
    loss='binary_crossentropy',
    metrics=['accuracy', 'binary_crossentropy']
)

rmsprop_model.compile(
    optimizer=keras.optimizers.RMSprop(lr=0.0001),
    loss='binary_crossentropy',
    metrics=['accuracy', 'binary_crossentropy']
)

sgd_model.compile(
    optimizer=keras.optimizers.SGD(lr=0.1, momentum=0.1, nesterov=True),
    loss='binary_crossentropy',
    metrics=['accuracy', 'binary_crossentropy']
)


"""
Train optimizer models
"""
# Optimization models
adam_history = adam_model.fit(train_data, train_labels, epochs=20, batch_size=500,
                                   validation_data=(test_data, test_labels), verbose=2)

rmsprop_history = rmsprop_model.fit(train_data, train_labels, epochs=20, batch_size=500,
                                        validation_data=(test_data, test_labels), verbose=2)

sgd_history = sgd_model.fit(train_data, train_labels, epochs=20, batch_size=500,
                                        validation_data=(test_data, test_labels), verbose=2)


# Visualize each models with different Optimizers
Visualize([('Adam', adam_history), ('RMSprop', rmsprop_history), ('SGD', sgd_history)])
