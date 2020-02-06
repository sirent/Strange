import re

import gensim
import numpy as np
import pandas as pd
from keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Embedding, Dropout
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split


###################################################
def remove_stopwords(data):
    stop_words = set(stopwords.words('indonesian'))
    wordss = []
    for sentence in data:
        sentence = sentence.lower().split()
        # print(sentence)
        x = [word.strip() for word in sentence if word not in stop_words]
        # print(x)
        wordss.append(x)
    return wordss


def remove_punc_emoji(data):
    # REMOVE EMOJI
    preprocess = [re.sub(r'[^\x00-\x7F]+', '', sentence) for sentence in data]
    # REMOVE ?
    preprocess2 = [re.sub(r'\?+', '', sentence) for sentence in preprocess]
    # REMOVE .
    preprocess3 = [re.sub(r'\.+', ' ', sentence) for sentence in preprocess2]
    # REMOVE HASHTAG
    preprocess4 = [re.sub(r'(#[^\s]+)+', '', sentence) for sentence in preprocess3]
    # REMOVE ' '
    preprocess5 = [re.sub(r'', '', sentence) for sentence in preprocess4]
    # REMOVE ,
    preprocess6 = [re.sub(r',', ' ', sentence) for sentence in preprocess5]

    return preprocess6


###################################################

# OPEN TRAIN DATA USING PANDAS
train_data = pd.read_csv("Training_Jokowimundurlahbatch1.csv", sep=';', encoding='latin-1')
print(train_data.head())
print("Total rows: {0}".format(len(train_data)))
print(list(train_data))

# OPEN TEST DATA USING PANDAS
test_data = pd.read_csv('Test_data.csv', sep=';', encoding='latin-1')
print(test_data.head())
print("Total rows: {0}".format(len(test_data)))
print(list(test_data))

# Comment in train data
print("\nComments")
print("-----------")
Train_Comments = train_data.Comment
print(Train_Comments)
# print(row[0])

# Comment in test data
print("\nComments")
print("-----------")
Test_Comments = test_data.Comment
print(Test_Comments)

# Labels in train data
print("\nLabels")
print("-----------")
# labels = ['Label']
# train_labels = train_data[labels]

# train_labels = train_data.Label

labels = train_data.Label.unique()
dic = {}
for i, Label in enumerate(labels):
    dic[Label] = i
train_labels = train_data.Label.apply(lambda x: dic[x])
print(train_labels)

# PREPROCESSING TRAINING DATA
preprocessed_train_comment = remove_punc_emoji(Train_Comments)
print(preprocessed_train_comment)

# PREPROCESSING TEST DATA
preprocessed_test_comment = remove_punc_emoji(Test_Comments)
print(preprocessed_test_comment)

# LOWER AND SPLIT SENTENCE TO WORDS ALSO REMOVE STOPWORDS
train_words = remove_stopwords(preprocessed_train_comment)  # TRAINING DATA
test_words = remove_stopwords(preprocessed_test_comment)  # TEST DATA

# INPUT TO WORD2VEC(SIZE = DIMENSION, MIN_COUNT = MINIMAL NUMBER OF WORDS APPEAR WILL BE LEARNED, ITER = ITERATION OF
# SENTENCE WILL BE LEARNED)
Embedding_dim = 300
train_word_model = gensim.models.Word2Vec(train_words, size=Embedding_dim, min_count=1, iter=10, sg=1, window=3)

# CHECK SIMILARITY WORD BANGSAT
# print(train_word_model.wv.most_similar(positive='bangsat'))

# EMBED WORD2VEC VOCABULARY TO MATRIX

# MAX_NB_WORDS = 100000
# nb_words = min(MAX_NB_WORDS, len(train_word_model.wv.vocab))+1

embedding_matrix = np.zeros((len(train_word_model.wv.vocab) + 1, Embedding_dim))
# embedding_matrix = np.zeros((nb_words, Embedding_dim))
for i, vec in enumerate(train_word_model.wv.vectors):
    embedding_matrix[i] = vec
print(train_word_model.wv.vocab)


# how many features should the tokenizer extract
features = 500
tokenizer = Tokenizer(num_words=features)

# fit the tokenizer on our text
tokenizer.fit_on_texts(train_words)

# get all words that the tokenizer knows
word_index = tokenizer.word_index

# put the tokens in a matrix
X = tokenizer.texts_to_sequences(train_words)
X = pad_sequences(X)

# prepare the labels
y = pd.get_dummies(train_labels)

# split in train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, shuffle=False)

########################################################################################################################
model = Sequential()
model.add(
    Embedding(len(train_word_model.wv.vocab) + 1, Embedding_dim, input_length=X.shape[1], weights=[embedding_matrix],
              trainable=False))

model.add(Dropout(0.2))

model.add(Conv1D(300,3,padding='valid',activation='relu',strides=2))
model.add(Conv1D(150,3,padding='valid',activation='relu',strides=2))
model.add(Conv1D(75,3,padding='valid',activation='relu',strides=2))

# model.add(Conv1D(filters=16, kernel_size=3, activation='relu'))
# model.add(Conv1D(filters=16, kernel_size=4, activation='relu'))
# model.add(Conv1D(filters=16, kernel_size=5, activation='relu'))
# model.add(Conv1D(filters=16, kernel_size=8, activation='relu'))

# model.add(Conv1D(filters=32, kernel_size=3, activation='relu'))
# model.add(Conv1D(filters=32, kernel_size=4, activation='relu'))
# model.add(Conv1D(filters=32, kernel_size=5, activation='relu'))
# model.add(Conv1D(filters=32, kernel_size=8, activation='relu'))

# model.add(Conv2D(filters=100, kernel_size=8, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
# model.add(Conv2D(filters=100, kernel_size=4, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
# model.add(Conv2D(filters=100, kernel_size=5, activation='relu', kernel_regularizer=regularizers.l2(0.01)))

model.add(MaxPooling1D(pool_size=2))
# model.add(MaxPooling2D(pool_size=2))

model.add(Flatten())
model.add(Dropout(0.2))
# model.add(Dense(10, activation='relu'))

model.add(Dense(150, activation='sigmoid'))

model.add(Dropout(0.2))
# model.add(Dense(1, activation='sigmoid'))

model.add(Dense(3, activation='sigmoid'))

# model.add(Dense(y.shape[1], activation="softmax"))
model.add(Dense(y.shape[1], activation="sigmoid"))

print(model.summary())
# model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=['acc'])
model.compile(optimizer="rmsprop", loss="categorical_crossentropy", metrics=['acc'])
# model.compile(loss='mean_squared_error', optimizer='sgd', metrics=['mae', 'acc'])
#############################################

# EVALUATION TEST
batch = 128
epo = 2
model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=batch, epochs=epo)

evaluate = model.evaluate(X_test, y_test, verbose=0)
print('\nTest Loss:', evaluate[0])
print('Test Accuracy:', evaluate[1])
###############################################

# REAL DATA TEST
sequences_test = tokenizer.texts_to_sequences(test_data)
X_real_test = pad_sequences(sequences_test, maxlen=X_train.shape[1])
# X_real_test = pad_sequences(sequences_test, maxlen=MAX_SEQUENCE_LENGTH)
y_pred = model.predict(X_real_test)
to_submit = pd.DataFrame(index=test_data.Id, data={'Label': y_pred[:, dic['Label']]})
to_submit.to_csv('submit.csv')
##############################################
