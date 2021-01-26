import json
import tensorflow as tf
import numpy as np
import urllib
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras.callbacks import EarlyStopping
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

def solution_model():
    # Download the zip file once, then load from local for constant debugging

    # url = 'https://storage.googleapis.com/download.tensorflow.org/data/sarcasm.json'
    # urllib.request.urlretrieve(url, 'sarcasm.json')
    with open('sarcasm.json') as file:
        data = json.load(file)

    # DO NOT CHANGE THIS CODE
    vocab_size = 1000
    embedding_dim = 16
    max_length = 120
    trunc_type='post'
    padding_type='post'
    oov_tok = "<OOV>"
    training_size = 20000

    sentences = []
    labels = []

    dataset = []
    for elem in data:
        sentences.append(elem['headline'])
        labels.append(elem['is_sarcastic'])

    # Train test split
    training_size = int(len(data) * 0.2)
    train_sentences = sentences[:training_size]
    train_labels = labels[:training_size]
    validation_sentences = sentences[training_size:]
    validation_labels = labels[training_size:]

    train_labels = np.array(train_labels)
    validation_labels = np.array(validation_labels)

    # define Tokenizer
    tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)

    # fit tokenizer on text
    tokenizer.fit_on_texts(sentences)
    word_to_idx = tokenizer.word_index

    # Text to sequence mapping
    train_sequences = tokenizer.texts_to_sequences(train_sentences)
    validation_sequences = tokenizer.texts_to_sequences(validation_sentences)

    # Padding
    train_padded = pad_sequences(train_sequences, truncating=trunc_type, padding=padding_type, maxlen=max_length)
    validation_padded = pad_sequences(validation_sequences, padding=padding_type, maxlen=max_length)

    # Don't have to tokenize lables cuz it's numbers

    training_label_seq = np.array(train_labels)
    validation_label_seq = np.array(validation_labels)

    # Another Structure of a model can be:
    # model = tf.keras.Sequential([
    #     tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
    #     tf.keras.layers.GlobalAveragePooling1D(),
    #     tf.keras.layers.Dense(24, activation='relu'),
    #     tf.keras.layers.Dense(6, activation='softmax')
    # ])

    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),

        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
        tf.keras.layers.Dense(24, activation='relu'),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model, train_padded, training_label_seq, validation_padded, validation_label_seq



if __name__ == '__main__':
    model, train_padded, training_label_seq, validation_padded, validation_label_seq = solution_model()

    early_stopping = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=3)

    history = model.fit(
        train_padded, training_label_seq,
        epochs=5,
        validation_data=(validation_padded, validation_label_seq),
        verbose=2,
        callbacks=[early_stopping]
    )

    model.save("mymodel.h5")

