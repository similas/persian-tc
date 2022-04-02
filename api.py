import pickle
import pandas as pd
import numpy as np
from keras.layers import LSTM, Activation, Dropout, Dense, Input, Conv1D, MaxPooling1D, GlobalMaxPooling1D
from keras.layers.embeddings import Embedding
from keras.models import Model
import string
import re
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import keras
from flask import Flask, request

with open('nonverbal',encoding="utf-8") as f:
    stopwords = [list(map(str, line.split())) for line in f]
    stopwords = [word[0] for word in stopwords]

def remove_stopwords(text):
  return ' '.join([word for word in text.split() if word not in (stopwords)])

def remove_tags(string):
    result = re.sub('<.*?>','',string)
    result = re.sub('(@[A-Za-z0-9]+)', '', result)
    return result

with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

words_to_index = tokenizer.word_index

# START GLOVE PART
def read_glove_vector(glove_vec):
    with open(glove_vec, 'r', encoding='UTF-8') as f:
        word_to_vec_map = {}
        for line in f:
            w_line = line.split()
            curr_word = w_line[0]
            word_to_vec_map[curr_word] = np.array(w_line[1:], dtype=np.float64)
    return word_to_vec_map

word_to_vec_map = read_glove_vector('vectors.txt')

maxLen_g = 20

vocab_len = len(words_to_index)
embed_vector_len = word_to_vec_map['سلام'].shape[0]

emb_matrix = np.zeros((vocab_len, embed_vector_len))

for word, index in words_to_index.items():
  embedding_vector = word_to_vec_map.get(word)
  if embedding_vector is not None:
    emb_matrix[index, :] = embedding_vector

embedding_layer_glove = Embedding(input_dim=vocab_len,
                            output_dim=embed_vector_len,
                            input_length=maxLen_g,
                            weights = [emb_matrix],
                            trainable=False)

def persian_tc_glove(input_shape):
  X_indices = Input(input_shape)
  embeddings = embedding_layer_glove(X_indices)
  X = LSTM(128, return_sequences=True)(embeddings)
  X = Dropout(0.5)(X)
  X = LSTM(128, return_sequences=True)(X)
  X = Dropout(0.5)(X)
  X = LSTM(128, return_sequences=True)(X)
  X = Dropout(0.5)(X)
  X = LSTM(64)(X)
  X = Dense(3, activation='softmax')(X)
  model = Model(inputs=X_indices, outputs=X)
  return model

model_glove = persian_tc_glove(maxLen_g)
model_glove.load_weights('W_glove/W_glove')
# END GLOVE PART


# START W2V PART
with open('w2v_dict.p', 'rb') as fp:
    w2v_dict = pickle.load(fp)

word_to_vec_map = w2v_dict
maxLen_w2v = 15
vocab_len = len(words_to_index)
embed_vector_len = word_to_vec_map['سلام'].shape[0]

emb_matrix = np.zeros((vocab_len, embed_vector_len))

for word, index in words_to_index.items():
  try:
      embedding_vector = word_to_vec_map[word]
  except:
      embedding_vector = None
  if embedding_vector is not None:
    emb_matrix[index, :] = embedding_vector

embedding_layer = Embedding(input_dim=vocab_len,
                            output_dim=embed_vector_len,
                            input_length=maxLen_w2v,
                            weights = [emb_matrix],
                            trainable=False)

def persian_tc_w2v(input_shape):
  X_indices = Input(input_shape)
  embeddings = embedding_layer(X_indices)
  X = LSTM(128, return_sequences=True)(embeddings)
  X = Dropout(0.5)(X)
  X = LSTM(64)(X)
  X = Dense(3, activation='softmax')(X)
  model = Model(inputs=X_indices, outputs=X)
  return model

model_w2v = persian_tc_w2v(maxLen_w2v)
model_w2v.load_weights("W_w2v/W_w2v")

adam = keras.optimizers.adam_v2.Adam(learning_rate = 0.001)
model_w2v.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
model_glove.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])

app = Flask(__name__)

@app.route('/w2v', methods=['POST', 'GET'])
def api_w2v():
    if request.method == 'POST':
        text = request.form.get('text')
        text = remove_tags(remove_stopwords(text))
        text_indices = tokenizer.texts_to_sequences([text])
        to_predict = pad_sequences(text_indices, maxlen=maxLen_w2v, padding='post')
        negative, neutral, positive =  model_w2v.predict([to_predict])[0]
        return f"negative : {negative}, neutral : {neutral}, positive : {positive}"
    else:
        return "TEXT CLASSIFICATION API :)"

@app.route('/glove', methods=['POST', 'GET'])
def api_glove():
    if request.method == 'POST':
        text = request.form.get('text')
        text = remove_tags(remove_stopwords(text))
        text_indices = tokenizer.texts_to_sequences([text])
        to_predict = pad_sequences(text_indices, maxlen=maxLen_g, padding='post')
        negative, neutral, positive =  model_glove.predict([to_predict])[0]
        return f"negative : {negative}, neutral : {neutral}, positive : {positive}"
    else:
        return "TEXT CLASSIFICATION API :)"

if __name__ == '__main__':
    app.run(debug=True)
