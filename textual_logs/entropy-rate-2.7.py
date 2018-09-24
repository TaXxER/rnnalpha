import re, codecs, random, math, textwrap
from collections import defaultdict, deque, Counter

def tokenize(lines, tokenizer):
	for line in lines:
		for token in tokenizer(line.lower().strip()):
			yield token
				
def read(file_path):
    with codecs.open(file_path, mode="r", encoding="utf-8") as file:
        return [chomp(line) for line in file]
		
def chars(lines):
	return tokenize(lines, lambda s: ' ' + s)
	
def markov_model(stream, model_order):
	model, stats = defaultdict(Counter), Counter()
	circular_buffer = deque(maxlen = model_order)
	stream = list(stream)
	for i, token in enumerate(stream):
		prefix = tuple(circular_buffer)
		if len(prefix) <= model_order:
			model[prefix][token] += 1.0
			stats[prefix] += 1.0
		circular_buffer.append(token)
	return model, stats

def total_brier_score(model, testmodel, stats):
    return sum(brier_score(model, testmodel, stats, prefix) for prefix in stats) / (sum(stats.values()) * len(set.union(*[set(x) for x in set(stats.keys())]))-1) # -1 for the ' '-character

def brier_score(model, testmodel, stats, prefix):
    #if ' ' in prefix: # outcomment this if-statement for AKOM model
    #    return stats[prefix]
    actual = testmodel[prefix]
    n_prefix = prefix
    while n_prefix not in model.keys():
        n_prefix = n_prefix[1:]
    probas = model[n_prefix]
    normalization_factor = sum(probas.values())
    if normalization_factor == 0:
        return 1
    brier = 0
    for i, count in actual.items(): # predicted probabilities
        for j, count2 in probas.items(): # true probabilities
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

all_lines = read("sepsis.txt")
random.shuffle(all_lines)
elems_per_fold = int(round(len(all_lines)/3))
train = all_lines[:2*elems_per_fold]
test = all_lines[2*elems_per_fold:]

for i in range(1,10):
    model, stats = markov_model(chars(all_lines), i)
    model_test, stats_test = markov_model(chars(all_lines), i)
    print("Order: {}, Brier score: {}".format(i,total_brier_score(model, model_test, stats_test)))