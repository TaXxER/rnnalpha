from splearn.spectral import Spectral
from splearn.datasets.base import load_data_sample
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import train_test_split
import unicodecsv
import numpy as np
import random
import sys
import os
import copy
import csv

X = load_data_sample('spice_logs/sepsis.spice')

#random.shuffle(lines)

#elems_per_fold = int(round(numlines/3))
#fold1 = lines[:elems_per_fold]
#with open('output_files/folds/fold1.csv', 'w') as csvfile:
#    spamwriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
#    for row in fold1:
#        spamwriter.writerow([str(s).encode("utf-8") for s in row])

#fold2 = lines[elems_per_fold:2*elems_per_fold]
#with open('output_files/folds/fold2.csv', 'w') as csvfile:
#    spamwriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
#    for row in fold2:
#        spamwriter.writerow([str(s).encode("utf-8") for s in row])
        
#fold3 = lines[2*elems_per_fold:]
#with open('output_files/folds/fold3.csv', 'w') as csvfile:
#    spamwriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
#    for row in fold3:
#        spamwriter.writerow([str(s).encode("utf-8") for s in row])

#lines = fold1 + fold2
X_train, X_test = train_test_split(X, test_size=0.33)
# build the model: 
print('Build model...')
model = Spectral()
# train the model, output generated text after each iteration
model.fit(X.data)

# test model
#sentences = []
#next_chars = []
#for line in fold3:
#    for i in range(0, len(line), step):
#        sentences.append(line[0: i])
#        next_chars.append(line[i])

#X_test = np.zeros((len(sentences), maxlen, num_features), dtype=np.int32)
#y_test = np.zeros((len(sentences), len(target_chars)), dtype=np.float32)
#y_hat = np.empty((len(target_chars)), dtype=np.float32)
#for i, sentence in enumerate(sentences):
#    if(len(sentence)>maxlen):
#        sentence = sentence[0:maxlen] 
#    leftpad = maxlen-len(sentence)
#    for t, char in enumerate(sentence):
#        for c in chars:
#            if c==char:
#                X_test[i, t+leftpad, char_indices[c]] = 1
#        X_test[i, t+leftpad, len(chars)] = t+1

    #if next_chars[i]=='}':
    #    continue
#    for c in target_chars:
#        if c==next_chars[i]:
#            y_test[i, char_indices[c]] = 1-softness
#        else:
#            y_test[i, char_indices[c]] = softness/(len(target_chars)-1)
    #y_hat = np.vstack((y_hat,model.predict(X[i].reshape((1,maxlen,num_features)))))

#y_hat = model.predict_proba(X_test)

#print(np.mean(list(map(lambda x: brier_score_loss(y_test[x],y_hat[x]),[i[0] for i in enumerate(y_test)]))))