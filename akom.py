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

def brier_score(model, testmodel, stats, prefix):
    #if prefix not in model.keys(): # outcomment this if-statement for AKOM model
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


# for reproducibility
random.seed(22)

all_lines = read(argv[1])
n_vocabulary = len(set([c for seq in all_lines for c in seq])) + 1 # +1 is for the end symbol

all_brier_scores = []
for i in range(3):
    random.shuffle(all_lines)
    elems_per_fold = int(round(len(all_lines)/3))
    train = all_lines[:2*elems_per_fold]
    test = all_lines[2*elems_per_fold:]

    # model selection
    random.shuffle(train)
    n_val_traces = int(round(len(train) * 0.2))
    val_selection = train[:n_val_traces]
    train_selection = train[n_val_traces:]

    brier_scores_all_orders = {}
    for order in range(1,20):
        model, stats = markov_model(chars(train_selection), order)
        model_test, stats_test = markov_model(chars(val_selection), order)
        score_current_order = total_brier_score(model, model_test, stats_test)
        brier_scores_all_orders[order] = score_current_order
        print("[Validation] Iter: {}, Order: {}, Brier score: {}".format(i, order, score_current_order))

    # train and evaluate final model
    best_order = min(brier_scores_all_orders.items(), key=operator.itemgetter(1))[0]
    model, stats = markov_model(chars(train), best_order)
    model_test, stats_test = markov_model(chars(test), best_order)
    final_brier_score = total_brier_score(model, model_test, stats_test)
    print("[Test] Iter: {}, Order: {}, Brier score: {}".format(i, best_order, final_brier_score))
    print()
    
    all_brier_scores.append(final_brier_score)
    
print("Brier scores: ", all_brier_scores)