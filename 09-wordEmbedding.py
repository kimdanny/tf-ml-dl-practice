
"""
Very Simple Sentiment Classifier with single Embedding layer
"""
import numpy as np
from numpy import array
import tensorflow as tf
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers


def pad_document(encoded_docs, max_length):
    """
    Make sure all the encoded sentences have the max_length
    padding='post' option put 0s after the final embedding value
    """
    padded_docs = pad_sequences(encoded_docs, maxlen=max_length, padding='post')

    return padded_docs


def make_embedding_model(vocab_size, max_length):
    """
    tensorflow.keras.layers.Embedding:
        Turns positive integers (indexes) into dense vectors of fixed size.
        Embedding(input_dim, output_dim, input_length=None)
            :input_dim
                -> Integer. Size of the vocab
            :output_dim
                -> Integer. Dimension of the dense embedding
            :input_length
                -> Length of input sequences, when it is constant.
                   This argument is required if you are going to connect Flatten() then Dense() layers.
                   Without input_length, the shape of the dense outputs cannot be computed.

        e.g. [[4], [20]] -> [[0.25, 0.1], [0.6, -0.2]]

        This layer (Embedding layer) can only be used as the first layer in a model.
    """
    model = Sequential()
    # Embedding Layer. input_length is required as we connect the Flatten() and Dense() upstream
    model.add(layers.Embedding(vocab_size, 8, input_length=max_length))
    model.add(layers.Flatten())
    model.add(layers.Dense(1, activation=tf.nn.sigmoid))

    return model


docs = ['Well done!',
        'Good work',
        'Great effort',
        'nice work',
        'Excellent!',
        'Weak',
        'Poor effort!',
        'not good',
        'poor work',
        'Could have done better.']

""" <How docs changes during one-hot encoding and padding>
One-hot encoded: 
 [[34, 5], [30, 9], [6, 12], [24, 9], [5], [7], [15, 12], [41, 30], [15, 9], [11, 21, 5, 13]]

Padded docs: 
 [[34  5  0  0]
 [30  9  0  0]
 [ 6 12  0  0]
 [24  9  0  0]
 [ 5  0  0  0]
 [ 7  0  0  0]
 [15 12  0  0]
 [41 30  0  0]
 [15  9  0  0]
 [11 21  5 13]]

"""

# Sentiment labels
# set 1 for positives and 0 for negatives
labels = array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])

# Assume we have 50 vocab size
vocab_size = 50

# One sentence comprise of maximum of 4 words
max_length = 4

# one-hot vectorize the sentences in the docs
encoded_docs = [one_hot(sentence, vocab_size) for sentence in docs]
print("One-hot encoded: \n", encoded_docs)

# Make sure every sentence in the docs has the same lenght of max_length
padded_docs = pad_document(encoded_docs, max_length)
print("Padded docs: \n", padded_docs)

# Embedding and then Fatten followed by Dense(1) layer
model = make_embedding_model(vocab_size, max_length)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# Note that we train the model with encoded+padded input and labels
model.fit(padded_docs, labels, epochs=500)

# Evaluate the model
loss, accuracy = model.evaluate(padded_docs, labels)

print("Loss: {}, Accuracy: {}".format(loss,  accuracy))


"""
Little Example of Embedding layer from TF document:
https://www.tensorflow.org/api_docs/python/tf/keras/layers/Embedding
"""
del model
model = tf.keras.Sequential()
model.add(tf.keras.layers.Embedding(1000, 64, input_length=10))
# The model will take as input an integer matrix of size (batch, input_length),
# and the largest integer (i.e. word index) in the input should be no larger than 999 (vocabulary size).
# Now model.output_shape is (None, 10, 64), where `None` is the batch dimension.

input_array = np.random.randint(1000, size=(32, 10))
print(input_array)
# [[711 510 664 211 450 638 949 515 770 755]
#  [836 425 367 658 790 295 376  56  60  60]
#               ...
#               ...
#  [336 390 235 347  80  54 712 758 669 575]
#  [277 700 612 284 356 430 636 763 506 746]]

model.compile(optimizer='rmsprop', loss='mse')
output_array = model.predict(input_array)
print(output_array.shape)   # (32, 10, 64)

