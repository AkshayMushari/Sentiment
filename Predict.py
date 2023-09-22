import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
import pickle
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing import sequence
import nltk

# dataset is present on google drive


def predict(chats):

    max_len = 100
    pklfile = r"modelweights.pkl"
    model = keras.models.load_model(filepath='model.h5', compile=False)
    with open(pklfile, 'rb') as f:
        w = pickle.load(f)
        model.set_weights(w)

    with open('tokenizer.pickle', 'rb') as handle:
        tok = pickle.load(handle)
    sequences = tok.texts_to_sequences(chats)
    input_to_predict = sequence.pad_sequences(sequences, maxlen=max_len)
    input_to_predict = np.array(input_to_predict)
    positiveScore = 0
    negativeScore = 0
    output = model.predict(input_to_predict)
    print("##@$@#@ --> output: ", output)
    print('chats are ->>>>', chats)
    for i in output:
        if(i[0] > 0.5):
            positiveScore = positiveScore+1
        else:
            negativeScore = negativeScore+1
    # laplacian correction
    flag = False
    if(positiveScore <= 0 or negativeScore <= 0):
        flag = True
        positiveScore += 1
        negativeScore += 1
    totalScore = (positiveScore/(positiveScore+negativeScore))
    totalScore = round(totalScore, 2)
    if(flag):
        print('0 found  (Total, pos, neg)',
              (totalScore, positiveScore-1, negativeScore-1))
        return (totalScore, positiveScore-1, negativeScore-1)
    else:
        print('(Total , pos, neg)', (totalScore, positiveScore, negativeScore))
        return (totalScore, positiveScore, negativeScore)


# print(predict(chats=["Yes Yes Yes Yes YES!!!", "I am so so happy",
#       "Today is the best day", "This is just great"]))

# print(predict(chats=['I am so sad', "All the misfortunes are given to me", "This was a very bad day"
#       "I was in an accident and now I have to pay for my car's reapirs. I am already short on money and I am not sure if I will even be able to pay rent next week"]))
