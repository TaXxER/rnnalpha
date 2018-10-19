from splearn.spectral import Spectral
from splearn.datasets.base import load_data_sample
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import train_test_split
import numpy as np

# in debug mode the first trace of the test set is printed in detail
debug_mode = 0
X = load_data_sample('spice_logs/nasa_complete.spice')

X_train, X_test = train_test_split(X.data, test_size=(1/3))
# build the model: 
print('Build model...')
model = Spectral(full_svd_calculation=True)
model.fit(X_train)
# model._automaton.val calculates probabilities over complete sequences
# Ap.val calculates probabilities over prefixes
Ap = model._automaton.transformation(source = "classic", target = "prefix")
brier_scores = []
first_trace = 1

for test_seq in X_test:
    test_seq = [int(x) for x in test_seq]
    prefix = []
    if first_trace*debug_mode == 1:
        print(test_seq)
    for i, elem in enumerate(test_seq): # elem \in \{-1\}\union [1,|activities|]
        if first_trace*debug_mode == 1:
            print(prefix)
        # construct prediction vector for prefix
        y_hat = []
        if first_trace*debug_mode == 1:
            print('candidates:')
        for a in list(range(X.nbL-1)):
            #if a==0: # a \in [0,|activities], therefore, by mapping 0 to -1, a maps on elem.
            #    a = -1 
            candidate = prefix + [a+1]
            if first_trace*debug_mode == 1:
                print(candidate)
            #y_hat += [max(0, Ap.val(candidate))]
            y_hat += [Ap.val(candidate)]
            #print(max(0, Ap.val(candidate)))
        # add probability of sequence end to y_hat
        y_hat += [model._automaton.val(prefix)]
        #if first_trace*debug_mode == 1:
        #    print(y_hat)
        # scale to at least 0
        if min(y_hat)<0:
            y_hat = list(map(lambda x : x-min(y_hat), y_hat))
        #if first_trace*debug_mode == 1:
        #    print(y_hat)
        #y_hat += [0]
        divisor = sum(y_hat)
        if divisor == 0:
            divisor = 1
        y_hat = [x/divisor for x in y_hat]
        # construct ground truth vector for prefix
        y = np.zeros(X.nbL)
        if elem==-1:
            y[elem] = 1
        else:
            y[elem-1] = 1 # -1 to convert [1,activities] to [0,activities-1]
        brier_scores += [brier_score_loss(y,y_hat)]
        if first_trace*debug_mode==1:
            print(y)
            print(y_hat)
            print(brier_score_loss(y,y_hat))
            print()
        prefix += [elem]
        if elem == -1: # stop making predictions for this sequence after end symbol is observed 
            break
    if first_trace*debug_mode==1:
        print(np.sum(brier_scores)/len(brier_scores))
    first_trace = 0
print(len(brier_scores))
print(np.sum(brier_scores)/len(brier_scores))