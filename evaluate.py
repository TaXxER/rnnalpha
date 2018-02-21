from keras.models import load_model

csvfile = open('output_files/folds/fold3.csv', 'r')
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
    
load_model('model_1386-0.49.h5')

