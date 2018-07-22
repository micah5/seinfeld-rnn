from __future__ import print_function

from argparse import ArgumentParser

from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.utils.data_utils import get_file

import numpy as np
import random
import sys
import os
import io

parser = ArgumentParser()
parser.add_argument("-f", "--floyd", dest="floyd", default=False, action='store_true',
                    help="changes path of output to /output")
parser.add_argument("-p", "--predict", dest="predict", default=False, action='store_true',
                    help="predicts from model. if floyd argument true, uses path /model to find.")
parser.add_argument("-in", "--intensity", dest="intensity",
                    help="only predicts at a certain intensity")
parser.add_argument("-i", "--iterations", dest="iterations", type=int, default=10,
                    help="number of iterations to run")

args = parser.parse_args()
d = vars(args)
FLOYD, PREDICT, INTENSITY, ITERATIONS = d['floyd'], d['predict'], d['intensity'], d['iterations']
if PREDICT == True:
    ITERATIONS = 2

path = 'seinfeld_modified.txt'
with io.open(path, encoding='utf-8') as f:
    text = f.read().lower()

print('corpus length:', len(text))

words = set(text.split())
print('words', words)
words = sorted(words)
print('sorted words', words)

print("words",type(words))
print("total number of unique words",len(words))


word_indices = dict((c, i) for i, c in enumerate(words))
indices_word = dict((i, c) for i, c in enumerate(words))

print("word_indices", type(word_indices), "length:",len(word_indices) )
print("indices_words", type(indices_word), "length", len(indices_word))

maxlen = 30
step = 3
print("maxlen:",maxlen,"step:", step)
sentences = []
next_words = []
next_words= []
sentences1 = []
list_words = []

sentences2=[]
list_words=text.lower().split()

for i in range(0,len(list_words)-maxlen, step):
    sentences2 = ' '.join(list_words[i: i + maxlen])
    sentences.append(sentences2)
    next_words.append((list_words[i + maxlen]))
print('nb sequences(length of sentences):', len(sentences))
print("length of next_word",len(next_words))

print('Vectorization...')
X = np.zeros((len(sentences), maxlen, len(words)), dtype=np.bool)
y = np.zeros((len(sentences), len(words)), dtype=np.bool)
for i, sentence in enumerate(sentences):
    for t, word in enumerate(sentence.split()):
        #print(i,t,word)
        X[i, t, word_indices[word]] = 1
    y[i, word_indices[next_words[i]]] = 1


#build the model: 2 stacked LSTM
print('Build model...')
model = Sequential()
model.add(LSTM(512, return_sequences=True, input_shape=(maxlen, len(words))))
model.add(Dropout(0.2))
model.add(LSTM(512, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(len(words)))
#model.add(Dense(1000))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

#if os.path.isfile('seinfeld_weights'):
if PREDICT == True:
    if FLOYD == True:
        model.load_weights('/model/seinfeld_weights.hdf5')
    else:
        model.load_weights('seinfeld_weights.hdf5')

def sample(a, temperature=1.0):
    # helper function to sample an index from a probability array
    a = np.log(a) / temperature
    dist = np.exp(a)/np.sum(np.exp(a))
    choices = range(len(a))
    return np.random.choice(choices, p=dist)

# train the model, output generated text after each iteration
for iteration in range(1, ITERATIONS):
    print()
    print('-' * 50)
    print('Iteration', iteration)
    if PREDICT == False:
        model.fit(X, y, batch_size=128, nb_epoch=2)
        if FLOYD == True:
            model.save_weights('/output/seinfeld_weights.hdf5', overwrite=True)
        else:
            model.save_weights('seinfeld_weights%d.hdf5' % iteration, overwrite=True)

    start_index = random.randint(0, len(list_words) - maxlen - 1)

    arr = [0.2, 0.5, 1.0, 1.2]
    if INTENSITY != None:
        arr = [INTENSITY]
    for diversity in arr:
        print()
        print('----- diversity:', diversity)
        generated = ''
        #sentence = ['the', 'money', 'i', 'dont', 'even', '<newline>', '<newline>', 'want', 'the', 'money', 'i', 'just', 'once', 'i', 'would', 'like', 'to', 'hear', 'a', 'dry', 'cleaner', 'admit', 'that', 'something', 'was', 'their', 'fault', 'thats', 'what', 'i']
        sentence = list_words[start_index: start_index + maxlen]
        generated += ' '.join(sentence)
        print('----- Generating with seed: "' , sentence , '"')
        print()
        sys.stdout.write(generated)
        print()

        for i in range(1000):
            x = np.zeros((1, maxlen, len(words)))
            for t, word in enumerate(sentence):
                x[0, t, word_indices[word]] = 1.

            preds = model.predict(x, verbose=0)[0]
            next_index = sample(preds, diversity)
            next_word = indices_word[next_index]
            generated += next_word
            del sentence[0]
            sentence.append(next_word)
            sys.stdout.write(' ')
            sys.stdout.write(next_word)
            sys.stdout.flush()
        print()
#model.save_weights('weights')
