'''
It is recommended to run this script on GPU, as recurrent
networks are quite computationally intensive.
'''

from keras.models import Sequential
from keras.layers.core import Dense
from keras.layers import CuDNNLSTM, CuDNNGRU, SimpleRNN, Activation, Dropout
from keras import regularizers
from keras.optimizers import Nadam, Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.layers.normalization import BatchNormalization
from sklearn.metrics import brier_score_loss
import numpy as np
import random
import sys
import os
from sys import argv
import tensorflow as tf


from hyperopt import Trials, STATUS_OK, tpe, fmin, hp
import hyperopt
from hyperopt.pyll.base import scope
from hyperopt.pyll.stochastic import sample

import keras
print(keras.__version__)
print(tf.__version__)

filename = argv[1]
model_type = argv[2]

with open(filename, "r") as fin:
    traces = [x.strip() + ' ' for x in fin]
        
maxlen = max([len(x) for x in traces])
nb_epochs = 200
early_stopping_patience = 20

vocabulary = set([val for trace in traces for val in trace])
vocabulary = {key: idx for idx, key in enumerate(vocabulary)}
num_chars = len(vocabulary)
print('total chars:', num_chars)
print('maxlen:', maxlen)

def generate_vectorized_data(traces):
    prefixes = []
    target_chars = []
    for trace in traces:
        for i in range(len(trace)):
            prefixes.append(trace[:i])
            target_chars.append(trace[i])
    print('nb prefixes:', len(prefixes))

    print('Vectorization...')
    X = np.zeros((len(prefixes), maxlen, num_chars), dtype=np.int32)
    y = np.zeros((len(prefixes), num_chars), dtype=np.float32)
    for i, prefix in enumerate(prefixes):
        leftpad = maxlen - len(prefix)
        for t, char in enumerate(prefix):
            X[i, t+leftpad, vocabulary[char]] = 1
        y[i, vocabulary[target_chars[i]]] = 1
        
    return X, y
  
def get_model(params):
    model = Sequential()
    
    n_layers = int(params["n_layers"]["n_layers"])

    if model_type == "LSTM":
        model.add(CuDNNLSTM(int(params["lstmsize"]),
                               kernel_initializer='glorot_uniform',
                               return_sequences=(n_layers != 1),
                               kernel_regularizer=regularizers.l1_l2(params["l1"],params["l2"]),
                               recurrent_regularizer=regularizers.l1_l2(params["l1"],params["l2"]),
                               input_shape=(maxlen, num_chars)))
        model.add(Dropout(params["dropout"]))


        for i in range(2, n_layers+1):
            return_sequences = (i != n_layers)
            model.add(CuDNNLSTM(int(params["n_layers"]["lstmsize%s%s" % (n_layers, i)]),
                           kernel_initializer='glorot_uniform',
                           return_sequences=return_sequences,
                           kernel_regularizer=regularizers.l1_l2(params["l1"],params["l2"]),
                           recurrent_regularizer=regularizers.l1_l2(params["l1"],params["l2"])))
            model.add(Dropout(params["dropout"]))
            
    elif model_type == "GRU":
        model.add(CuDNNGRU(int(params["lstmsize"]),
                               kernel_initializer='glorot_uniform',
                               return_sequences=(n_layers != 1),
                               kernel_regularizer=regularizers.l1_l2(params["l1"],params["l2"]),
                               recurrent_regularizer=regularizers.l1_l2(params["l1"],params["l2"]),
                               input_shape=(maxlen, num_chars)))
        model.add(Dropout(params["dropout"]))


        for i in range(2, n_layers+1):
            return_sequences = (i != n_layers)
            model.add(CuDNNGRU(int(params["n_layers"]["lstmsize%s%s" % (n_layers, i)]),
                           kernel_initializer='glorot_uniform',
                           return_sequences=return_sequences,
                           kernel_regularizer=regularizers.l1_l2(params["l1"],params["l2"]),
                           recurrent_regularizer=regularizers.l1_l2(params["l1"],params["l2"])))
            model.add(Dropout(params["dropout"]))
            
    elif model_type == "RNN":
        model.add(SimpleRNN(int(params["lstmsize"]),
                               kernel_initializer='glorot_uniform',
                               return_sequences=(n_layers != 1),
                               kernel_regularizer=regularizers.l1_l2(params["l1"],params["l2"]),
                               recurrent_regularizer=regularizers.l1_l2(params["l1"],params["l2"])))
        model.add(Dropout(params["dropout"]))


        for i in range(2, n_layers+1):
            return_sequences = (i != n_layers)
            model.add(SimpleRNN(int(params["n_layers"]["lstmsize%s%s" % (n_layers, i)]),
                           kernel_initializer='glorot_uniform',
                           return_sequences=return_sequences,
                           kernel_regularizer=regularizers.l1_l2(params["l1"],params["l2"]),
                           recurrent_regularizer=regularizers.l1_l2(params["l1"],params["l2"])))
            model.add(Dropout(params["dropout"]))
    
    model.add(Dense(num_chars, kernel_initializer='glorot_uniform'))
    model.add(Activation(tf.nn.softmax))
    opt = Adam(lr=params["learning_rate"])
    model.compile(loss='mean_squared_error', optimizer=opt)

    return model

def train_and_evaluate_model(params):
    print(params)
    model = get_model(params)
    early_stopping = EarlyStopping(monitor='val_loss', patience=early_stopping_patience, restore_best_weights=True)
    lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=100, verbose=0, mode='auto', epsilon=0.0001, cooldown=0, min_lr=0)

    # train the model, output generated text after each iteration
    history = model.fit(X_train, y_train, 
              validation_split=0.2,
              callbacks=[early_stopping, lr_reducer],
              batch_size=2**params['batch_size'], epochs=nb_epochs, verbose=2)
    
    scores = [history.history['val_loss'][epoch] for epoch in range(len(history.history['loss']))]
    score = min(scores)
    print(score)
    global best_score, best_model
    if best_score > score:
        best_score = score
        best_model = model
    
    return {'loss': score, 'status': STATUS_OK}

    
random.seed(22) # for reproducibility


space = {'lstmsize': scope.int(hp.loguniform('lstmsize', np.log(10), np.log(150))),
         'dropout': hp.uniform("dropout", 0, 0.5),
         'l1': hp.loguniform("l1", np.log(0.00001), np.log(0.1)),
         'l2': hp.loguniform("l2", np.log(0.00001), np.log(0.1)),
         'batch_size': scope.int(hp.uniform('batch_size', 3, 6)),
         'learning_rate': hp.loguniform("learning_rate", np.log(0.00001), np.log(0.01)),
         'n_layers': hp.choice('n_layers', [
             {'n_layers': 1},
             {'n_layers': 2,
              'lstmsize22': scope.int(hp.loguniform('lstmsize22', np.log(10), np.log(150)))},
             {'n_layers': 3,
              'lstmsize32': scope.int(hp.loguniform('lstmsize32', np.log(10), np.log(150))),
              'lstmsize33': scope.int(hp.loguniform('lstmsize33', np.log(10), np.log(150)))}
         ])
        }
        
n_iter = 60

final_brier_scores = []
for _ in range(3):
    
    # split into train and test set
    random.shuffle(traces)
    elems_per_fold = int(round(len(traces)/3))
    train = traces[:2*elems_per_fold]
    test = traces[2*elems_per_fold:]
    
    # vectorize datasets for model selection
    X_train, y_train = generate_vectorized_data(train)
    
    print(X_train.shape, y_train.shape)
    
    # model selection
    print('Starting model selection...')
    best_score = np.inf
    best_model = None
    trials = Trials()
    best = fmin(train_and_evaluate_model, space, algo=tpe.suggest, max_evals=n_iter, trials=trials)
    best_params = hyperopt.space_eval(space, best)
    print(best_params)
    
    # vectorize datasets for final evaluation
    X_test, y_test = generate_vectorized_data(test)
    
    # evaluate
    print('Evaluating final model...')
    preds = best_model.predict(X_test)
    brier_score = np.mean(list(map(lambda x: brier_score_loss(y_test[x],preds[x]),[i[0] for i in enumerate(y_test)])))
    
    print(brier_score)
    final_brier_scores.append(brier_score)

print(final_brier_scores)