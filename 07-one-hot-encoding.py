"""
Word Embedding -> One-hot Encoding
"""
import tensorflow as tf
# from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.text import Tokenizer

sentence1 = "I am going to have chicken for dinner"
sentence2 = "Stock market is very bullish today and I love it"

sentence1, sentence2 = sentence1.split(' '), sentence2.split(' ')
# sentence1 = ['I', 'am', 'going', 'to', 'have', 'chicken', 'for', 'dinner']
# sentence2 = ['Stock', 'market', 'is', 'very', 'bullish', 'today', 'and', 'I', 'love', 'it']

tokenizer = Tokenizer()

# Fit tokenizer on two sentences and store the indices of each word in word dict
tokenizer.fit_on_texts(sentence1+sentence2)
word_dict = tokenizer.word_index
print(word_dict)
# {'i': 1, 'am': 2, 'going': 3, 'to': 4, 'have': 5, 'chicken': 6, 'for': 7, 'dinner': 8, 'stock': 9, 'market': 10, 'is': 11, 'very': 12, 'bullish': 13, 'today': 14, 'and': 15, 'love': 16, 'it': 17}

# TODO: Using Tokenizer.texts_to_sequence, change both sentence to a numerical sequence
sen1 = tokenizer.texts_to_sequences(sentence1)
# [[1], [2], [3], [4], [5], [6], [7], [8]]
sen2 = tokenizer.texts_to_sequences(sentence2)
# [[9], [10], [11], [12], [13], [14], [15], [1], [16], [17]]

sen1 = [token[0] for token in sen1]  # [1, 2, 3, 4, 5, 6, 7, 8]
sen2 = [token[0] for token in sen2]  # [9, 10, 11, 12, 13, 14, 15, 1, 16, 17]

onehot_sen1 = sum(tf.one_hot(sen1, len(word_dict)))
onehot_sen2 = sum(tf.one_hot(sen2, len(word_dict)))

print(onehot_sen1)  # tf.Tensor([0. 1. 1. 1. 1. 1. 1. 1. 1. 0. 0. 0. 0. 0. 0. 0. 0.], shape=(17,), dtype=float32)
print(onehot_sen2)  # tf.Tensor([0. 1. 0. 0. 0. 0. 0. 0. 0. 1. 1. 1. 1. 1. 1. 1. 1.], shape=(17,), dtype=float32)

# Cosine similarity of one hot encoded sentences
cos_simil = onehot_sen1 * onehot_sen2 / (tf.norm(onehot_sen1) * tf.norm(onehot_sen2))
print("cosine similarity of two sentences: ", cos_simil)

# What happens to the cosine similarity if length of word is increased?
len_word = 500_000

onehot_sen1 = sum(tf.one_hot(sen1, len_word))
onehot_sen2 = sum(tf.one_hot(sen2, len_word))
cos_simil = onehot_sen1 * onehot_sen2 / (tf.norm(onehot_sen1) * tf.norm(onehot_sen2))

print("Cosine similarity when word length is 500,000: ", cos_simil)
# decreases

