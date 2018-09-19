'''
It is recommended to run this script on GPU, as recurrent
networks are quite computationally intensive.
'''


from keras.models import Sequential
from keras.layers.core import Dense
from keras.layers import LSTM, CuDNNLSTM, CuDNNGRU, RNN, SimpleRNN
from keras.utils.data_utils import get_file
from keras import regularizers
from keras.optimizers import Nadam, Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.layers.normalization import BatchNormalization
from sklearn.metrics import brier_score_loss
import unicodecsv
import numpy as np
import random
import sys
import os
import copy
import csv

csvfile = open('data/bpi12.csv', 'r')
spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
next(spamreader, None)  # skip the headers
lastcase = ''
line = ''
lines = []
ascii_offset = 161
numlines = 0
for row in spamreader:
    if row[0]!=lastcase:
        lastcase = row[0]
        lines.append(line)
        line = ''
        numlines+=1
    line+=chr(int(row[1])+ascii_offset )

random.shuffle(lines)


elems_per_fold = int(round(numlines/3))
fold1 = lines[:elems_per_fold]
with open('output_files/folds/fold1.csv', 'w') as csvfile:
    spamwriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    for row in fold1:
        spamwriter.writerow([str(s).encode("utf-8") for s in row])

fold2 = lines[elems_per_fold:2*elems_per_fold]
with open('output_files/folds/fold2.csv', 'w') as csvfile:
    spamwriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    for row in fold2:
        spamwriter.writerow([str(s).encode("utf-8") for s in row])
        
fold3 = lines[2*elems_per_fold:]
with open('output_files/folds/fold3.csv', 'w') as csvfile:
    spamwriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    for row in fold3:
        spamwriter.writerow([str(s).encode("utf-8") for s in row])

lines = fold1 + fold2

#path = os.path.abspath("E:/event_logs/activity/moves/rnn/encoded.txt")#get_file('encoded_log_wil.txt', origin="E:/Git/sequence_modelling/encoded_log_wil.txt")
#text = open(path).read().lower()
#print('corpus length:', len(text))

# cut the text in semi-redundant sequences of maxlen characters
step = 1
sentences = []
softness = 0
next_chars = []
#lines = text.splitlines()
#lines = map(lambda x: '{'+x+'}',lines)
lines = [x+'!' for x in lines]
maxlen = max([len(x) for x in lines])

chars = [set(x) for x in lines]
chars = list(set().union(*chars))
chars.sort()
target_chars = copy.copy(chars)
#target_chars.remove('{')
print('total chars:', len(chars))
char_indices = dict((c, i) for i, c in enumerate(target_chars))
#char_indices['{'] = len(target_chars)
indices_char = dict((i, c) for i, c in enumerate(target_chars))
#indices_char[len(target_chars)] = '{'
#print(indices_char)

for line in lines:
    for i in range(0, len(line), step):
        sentences.append(line[0: i])
        next_chars.append(line[i])
print('nb sequences:', len(sentences))

print('Vectorization...')
num_features = len(chars)+1
X = np.zeros((len(sentences), maxlen, num_features), dtype=np.int32)
y = np.zeros((len(sentences), len(target_chars)), dtype=np.float32)
for i, sentence in enumerate(sentences):
    leftpad = maxlen-len(sentence)
    for t, char in enumerate(sentence):
        for c in chars:
            if c==char:
                X[i, t+leftpad, char_indices[c]] = 1
        X[i, t+leftpad, len(chars)] = t+1

    #if next_chars[i]=='}':
    #    continue
    for c in target_chars:
        if c==next_chars[i]:
            y[i, char_indices[c]] = 1-softness
        else:
            y[i, char_indices[c]] = softness/(len(target_chars)-1)
    
    #print(X[i,:,:])
    #print(y[i,:])

# build the model: 
print('Build model...')
model = Sequential()
#model.add(LSTM(1000, consume_less='gpu', init='glorot_uniform', return_sequences=True, dropout_W=0.4, input_shape=(maxlen, num_features)))
#model.add(BatchNormalization(axis=1))
#model.add(CuDNNLSTM(32, kernel_initializer='glorot_uniform', return_sequences=True, kernel_regularizer=regularizers.l1_l2(0.0001,0.0005), recurrent_regularizer=regularizers.l1_l2(0.0001,0.0005), input_shape=(maxlen, num_features)))
#model.add(CuDNNLSTM(32, kernel_initializer='glorot_uniform', return_sequences=True, kernel_regularizer=regularizers.l1_l2(0.0001,0.0005), recurrent_regularizer=regularizers.l1_l2(0.0001,0.0005), input_shape=(maxlen, num_features)))
#model.add(CuDNNLSTM(16, kernel_initializer='glorot_uniform', return_sequences=False, kernel_regularizer=regularizers.l1_l2(0.0001,0.0005), recurrent_regularizer=regularizers.l1_l2(0.0001,0.0005), input_shape=(maxlen, num_features)))
#model.add(SimpleRNN(32, kernel_initializer='glorot_uniform', return_sequences=True, kernel_regularizer=regularizers.l1_l2(0.0001,0.0005), recurrent_regularizer=regularizers.l1_l2(0.0001,0.0005), input_shape=(maxlen, num_features)))
#model.add(SimpleRNN(32, kernel_initializer='glorot_uniform', return_sequences=True, kernel_regularizer=regularizers.l1_l2(0.0001,0.0005), recurrent_regularizer=regularizers.l1_l2(0.0001,0.0005), input_shape=(maxlen, num_features)))
#model.add(SimpleRNN(64, kernel_initializer='glorot_uniform', return_sequences=False, kernel_regularizer=regularizers.l1_l2(0.0001,0.0005), recurrent_regularizer=regularizers.l1_l2(0.0001,0.0005), input_shape=(maxlen, num_features)))
#model.add(CuDNNLSTM(32, kernel_initializer='glorot_uniform', return_sequences=True, kernel_regularizer=regularizers.l1_l2(0.0001,0.0005), recurrent_regularizer=regularizers.l1_l2(0.0001,0.0005), input_shape=(maxlen, num_features)))
model.add(CuDNNGRU(64, kernel_initializer='glorot_uniform', return_sequences=False, kernel_regularizer=regularizers.l1_l2(0.0001,0.0005), recurrent_regularizer=regularizers.l1_l2(0.0001,0.0005), input_shape=(maxlen, num_features)))
#model.add(BatchNormalization(axis=1))
#model.add(LSTM(100, consume_less='gpu', init='glorot_uniform', return_sequences=True, W_regularizer=WeightRegularizer(l2=0.0005, l1=0.0001), input_shape=(maxlen, num_features)))
#model.add(BatchNormalization(axis=1))
#model.add(LSTM(100, consume_less='gpu', init='glorot_uniform', return_sequences=True, W_regularizer=WeightRegularizer(l2=0.0005, l1=0.0001), input_shape=(maxlen, num_features)))
#model.add(BatchNormalization(axis=1))
#model.add(LSTM(100, consume_less='gpu', init='glorot_uniform', return_sequences=False, W_regularizer=WeightRegularizer(l2=0.0005, l1=0.0001), input_shape=(maxlen, num_features)))
#model.add(BatchNormalization(axis=1))
#model.add(LSTM(100, consume_less='gpu', init='glorot_uniform', return_sequences=False, W_regularizer=WeightRegularizer(l1=0.00005), input_shape=(maxlen, num_features)))
#model.add(Dense(len(target_chars), activation='relu', init='glorot_uniform', W_regularizer=l1))
model.add(Dense(len(target_chars), activation='softmax', kernel_initializer='glorot_uniform'))

#opt = Nadam(lr=0.0002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004, clipvalue=2)
opt = Adam()

model.compile(loss='categorical_crossentropy', optimizer=opt)
early_stopping = EarlyStopping(monitor='val_loss', patience=201)
model_checkpoint = ModelCheckpoint('output_files/models/model_{epoch:02d}-{val_loss:.2f}.h5', monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto')
lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=100, verbose=0, mode='auto', epsilon=0.0001, cooldown=0, min_lr=0)

# train the model, output generated text after each iteration
model.fit(X, y, validation_split=0.2, callbacks=[early_stopping, model_checkpoint, lr_reducer], batch_size=maxlen, epochs=125)

# test model
sentences = []
next_chars = []
for line in fold3:
    for i in range(0, len(line), step):
        sentences.append(line[0: i])
        next_chars.append(line[i])

X_test = np.zeros((len(sentences), maxlen, num_features), dtype=np.int32)
y_test = np.zeros((len(sentences), len(target_chars)), dtype=np.float32)
#y_hat = np.empty((len(target_chars)), dtype=np.float32)
for i, sentence in enumerate(sentences):
    if(len(sentence)>maxlen):
        sentence = sentence[0:maxlen] 
    leftpad = maxlen-len(sentence)
    for t, char in enumerate(sentence):
        for c in chars:
            if c==char:
                X_test[i, t+leftpad, char_indices[c]] = 1
        X_test[i, t+leftpad, len(chars)] = t+1

    #if next_chars[i]=='}':
    #    continue
    for c in target_chars:
        if c==next_chars[i]:
            y_test[i, char_indices[c]] = 1-softness
        else:
            y_test[i, char_indices[c]] = softness/(len(target_chars)-1)
    #y_hat = np.vstack((y_hat,model.predict(X[i].reshape((1,maxlen,num_features)))))

y_hat = model.predict(X_test)

print(np.mean(list(map(lambda x: brier_score_loss(y_test[x],y_hat[x]),[i[0] for i in enumerate(y_test)]))))