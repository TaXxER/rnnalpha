import re, codecs, random, math, textwrap
from collections import defaultdict, deque, Counter
import operator
from sys import argv


def tokenize(lines, tokenizer):
	for line in lines:
		for token in tokenizer(line.lower().strip()):
			yield token
				
def read(file_path):
    with codecs.open(file_path, mode="r", encoding="utf-8") as file:
        return [chomp(line) for line in file]
		
def chars(lines):
	return tokenize(lines, lambda s: s + ' ')
	
def markov_model(stream, model_order):
	model, stats = defaultdict(Counter), Counter()
	circular_buffer = deque(maxlen = model_order)
	stream = list(stream)
	for i, token in enumerate(stream):
		prefix = tuple(circular_buffer)
		if len(prefix) <= model_order:
			model[prefix][token] += 1.0
			stats[prefix] += 1.0
		if token == ' ':
			circular_buffer = deque(maxlen = model_order)
		else:
			circular_buffer.append(token)
	return model, stats

def total_brier_score(model, testmodel, stats):
    return sum(brier_score(model, testmodel, stats, prefix) for prefix in stats) / (sum(stats.values()) * n_vocabulary)

def brier_score(model, testmodel, teststats, prefix):
    if prefix not in model.keys(): # outcomment this if-statement for AKOM model
        return teststats[prefix]
    actual = testmodel[prefix]
    n_prefix = prefix
    while n_prefix not in model.keys():
        n_prefix = n_prefix[1:]
    probas = model[n_prefix]
    normalization_factor = sum(probas.values())
    if normalization_factor == 0:
        return 1
    brier = 0
    for i, count in actual.items(): # true probabilities
        for j, count2 in probas.items(): # predicted probabilities
            proba = count2/normalization_factor
            if i==j:
                brier += count * math.pow(1-(proba),2)
            else:
                brier += count * math.pow(0-(proba),2)
    return brier
    
def chomp(x):
    if x.endswith("\r\n"): return x[:-2]
    if x.endswith("\n"): return x[:-1]
    return x
 
def pick(counter):
	sample, accumulator = None, 0
	for key, count in counter.items():
		accumulator += count
		if random.randint(0, accumulator - 1) < count:
			sample = key
	return sample
	
def generate(model, state, length):
	for token_id in range(0, length):
		yield state[0]
		state = state[1:] + (pick(model[state]), ) 

if "sepsis" in argv[1]:
    dataset = "Sepsis"
elif "bpi" in argv[1]:
    dataset = "BPI12"
elif "receipt" in argv[1]:
    dataset = "WABO receipt phase"
else:
    dataset = argv[1]

# for reproducibility
random.seed(22)

all_lines = read(argv[1])
n_vocabulary = len(set([c for seq in all_lines for c in seq])) + 1 # +1 is for the end symbol

for order in range(1,3):
    all_brier_scores = []
    for i in range(3):
        random.shuffle(all_lines)
        elems_per_fold = int(round(len(all_lines)/3))
        train = all_lines[:2*elems_per_fold]
        test = all_lines[2*elems_per_fold:]

        model, stats = markov_model(chars(train), order)
        model_test, stats_test = markov_model(chars(test), order)
        final_brier_score = total_brier_score(model, model_test, stats_test)

        all_brier_scores.append(final_brier_score)
    
    for score in all_brier_scores:
        print("%s,Order-%s Markov,,%s," % (dataset, order, score))
    print()
