import csv
import tensorflow as tf
from tensorflow.python.keras.callbacks import EarlyStopping
import numpy as np
import pandas as pd
import urllib
import ssl
ssl._create_default_https_context = ssl._create_unverified_context


def windowed_dataset(series, window_size, batch_size, shuffle_buffer):
    series = tf.expand_dims(series, axis=-1)
    ds = tf.data.Dataset.from_tensor_slices(series)
    ds = ds.window(window_size + 1, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda w: w.batch(window_size + 1))
    ds = ds.shuffle(shuffle_buffer)
    ds = ds.map(lambda w: (w[:-1], w[1:]))
    return ds.batch(batch_size).prefetch(1)


def solution_model():
    urllib.request.urlretrieve('https://storage.googleapis.com/download.tensorflow.org/data/Sunspots.csv',
                               'sunspots.csv')
    sunspots = pd.read_csv('sunspots.csv', sep=",")
    dataframe = sunspots[['Date', 'Monthly Mean Total Sunspot Number']]

    # Your data should be loaded into 2 Python lists called time_step
    # and sunspots. They are decleared here.
    time_step = []
    sunspots = []
    for idx, sunspot in dataframe.iterrows():
        time_step.append(sunspot[0])
        sunspots.append(sunspot[1])

    sunspots = np.array(sunspots)
    time_step = np.array(time_step)

    # Split the dataset into training and validation
    split_time = 3000
    time_train = time_step[:split_time]
    x_train = sunspots[:split_time]
    time_valid = time_step[split_time:]
    x_valid = sunspots[split_time:]

    window_size = 30
    batch_size = 32
    shuffle_buffer_size = 1000

    tf.keras.backend.clear_session()
    # Can use any random seed
    tf.random.set_seed(51)
    np.random.seed(51)
    train_set = windowed_dataset(x_train, window_size=window_size, batch_size=batch_size,
                                 shuffle_buffer=shuffle_buffer_size)

    model = tf.keras.models.Sequential([
        tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=-1), input_shape=[None]),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, return_sequences=True)),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
        tf.keras.layers.Dense(1),
        # The data is not normalized, so this lambda layer helps keep the MAE in line with expectations.
        tf.keras.layers.Lambda(lambda x: x * 400)
    ])

    lr_schedule = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 1e-8 * 10 ** (epoch / 20))
    optimizer = tf.keras.optimizers.SGD(lr=1e-8, momentum=0.9)
    model.compile(loss=tf.keras.losses.Huber(),
                  optimizer=optimizer,
                  metrics=["mae"])

    return model, train_set, lr_schedule


if __name__ == '__main__':
    model, train_set, lr_schedule = solution_model()
    early_stopping = EarlyStopping(monitor='loss', mode='min', verbose=1, patience=4)

    history = model.fit(train_set, epochs=100, callbacks=[lr_schedule, early_stopping])

    model.save("sunspot_model.h5")

