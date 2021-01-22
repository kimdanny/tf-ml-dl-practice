"""
Word2Vec changes words to vectors, and uses CBOW(Continuous Bag of Words) or Skip-Gram model.
CBOW predicts Target Words based on Context Word.
"""

import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras import utils


def tokenize(corpus):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(corpus)
    corpus_tokenized = tokenizer.texts_to_sequences(corpus)
    V = len(tokenizer.word_index)
    return corpus_tokenized, V


def initialize(V, N):
    np.random.seed(432)
    W1 = np.random.rand(V, N)
    W2 = np.random.rand(N, V)
    return W1, W2


def corpus2io(corpus_tokenized, V, window_size):
    for words in corpus_tokenized:
        L = len(words)
        for index, word in enumerate(words):
            contexts = []
            center = []
            s = index - window_size
            e = index + window_size + 1
            contexts = contexts + [words[i] - 1 for i in range(s, e) if 0 <= i < L and i != index]
            center.append(word - 1)
            x = utils.to_categorical(contexts, V)
            y = utils.to_categorical(center, V)
            yield (x, y)


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


class CBOW:
    def __init__(self, method='cbow', window_size=1, n_hidden=2, n_epochs=1, corpus='', learning_rate=0.1):
        self.window = window_size
        self.N = n_hidden
        self.n_epochs = n_epochs
        self.corpus = [corpus]
        self.eta = learning_rate
        self.method = self.cbow

    def cbow(self, context, label, W1, W2, loss):
        x = np.matrix(np.mean(context, axis=0))
        h = np.matmul(W1.T, x.T)
        u = np.matmul(W2.T, h)
        y_pred = softmax(u)
        e = -label.T + y_pred
        # outer product of h and e
        dW2 = np.outer(h, e)
        # outer product of h transpose and matmul of W2 and e
        dW1 = np.outer(h.T, np.matmul(W2, e))
        # gradient update
        new_W1 = W1 - self.eta * dW1
        new_W2 = W2 - self.eta * dW2
        loss += -float(u[label.T == 1]) + np.log(np.sum(np.exp(u)))
        return new_W1, new_W2, loss

    def predict(self, x, W1, W2):
        h = np.mean([np.matmul(W1.T, _x) for _x in x], axis=0)
        u = np.dot(W2.T, h)
        return softmax(u)

    def run(self):
        if len(self.corpus) == 0:
            raise ValueError('You need to specify a corpus of text.')

        corpus_tokenized, V = tokenize(self.corpus)
        # initialising weights
        W1, W2 = initialize(V, self.N)

        loss_vs_epoch = []
        for e in range(self.n_epochs):
            loss = 0.
            for context, center in corpus2io(corpus_tokenized, V, self.window):
                W1, W2, loss = self.method(context, center, W1, W2, loss)
            loss_vs_epoch.append(loss)

        return W1, W2, loss_vs_epoch


def adapt_corpus(x, corpus):
    corpus = corpus.split()
    idx = []
    for i in x:
        idx.append(corpus.index(i))
    results = []
    for j in idx:
        results.append([1 if i == j else 0 for i in range(len(corpus))])

    return results


def get_result(y, corpus):
    corpus = corpus.split()
    y = list(y)
    max_y = max(y)
    argmax_y = y.index(max_y)
    return corpus[argmax_y]


def main():
    corpus = "I like playing football with my friends."
    cbow = CBOW(method="cbow", corpus=corpus, window_size=1, n_hidden=2, n_epochs=10, learning_rate=0.8)

    W1, W2, loss_vs_epoch = cbow.run()

    x = ["I"]
    y_pred = cbow.predict(adapt_corpus(x, corpus), W1, W2)
    print("Prediction: ", get_result(y_pred, corpus))

    x = ["like"]
    y_pred = cbow.predict(adapt_corpus(x, corpus), W1, W2)
    print("Prediction: ", get_result(y_pred, corpus))


if __name__ == "__main__":
    main()
