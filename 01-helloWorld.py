import tensorflow as tf
from tensorflow import keras
import numpy as np

layer_0 = keras.layers.Dense(units=1, input_shape=[1])
model = keras.Sequential([layer_0])

model.compile(optimizer='sgd', loss='mean_squared_error')

xs = np.array([-1.0,  0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)

model.fit(xs, ys, epochs=300)

print(model.predict([10.0]))
print(f"Layer variables look like this: {layer_0.get_weights()}")
