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
	return tokenize(lines, lambda s: s + "")
	
def words(lines):
	return tokenize(lines, lambda s: re.findall(r"[a-zA-Z']+", s))

def markov_model(stream, model_order):
	model, stats = defaultdict(Counter), Counter()
	circular_buffer = deque(maxlen = model_order)
	stream = list(stream)
	for i, token in enumerate(stream):
		prefix = tuple(circular_buffer)
		if len(prefix) <= model_order:
			model[prefix][token] += 1.0
			if token != ' ':
			    stats[prefix] += 1.0
		circular_buffer.append(token)
	return model, stats

def entropy(stats, normalization_factor):
	return -sum(proba / normalization_factor * math.log(proba / normalization_factor, 2) for proba in stats.values())

def entropy_rate(model, stats):
	return sum(stats[prefix] * entropy(model[prefix], stats[prefix]) for prefix in stats) / sum(stats.values())

def total_brier_score(model, stats):
    #print(sum(brier_score(model[prefix], stats[prefix]) for prefix in stats))
    #print(math.pow(sum(stats.values()),2))
    #print()
    return math.sqrt(sum(brier_score(model[prefix], stats[prefix]) for prefix in stats) / sum(stats.values()) * math.pow(len(set.union(*[set(x) for x in set(stats.keys())])),2))

def total_brier_score2(model, testmodel, stats):
    #print(sum(brier_score(model[prefix], stats[prefix]) for prefix in stats))
    #print(math.pow(sum(stats.values()),2))
    #print()
    #print(len(set.union(*[set(x) for x in set(stats.keys())])))
    # number of activities = len(set.union(*[set(x) for x in set(stats.keys())])) 
    return math.sqrt(sum(brier_score2(model, testmodel, stats, prefix) for prefix in stats) / (sum(stats.values()) * math.pow(len(set.union(*[set(x) for x in set(stats.keys())])),2) ))


def brier_score2(model, testmodel, stats, prefix):
    normalization_factor = stats[prefix]
    #print('total times prefix seen: {}'.format(normalization_factor))
    actual = testmodel[prefix]
    o_prefix = prefix
    while prefix not in model:
        prefix = prefix[:-1]
    probas = model[prefix]
    #print(probas)
    #print()
    normalization_factor2 = sum(probas.values())
    #print(normalization_factor2)
    #print(normalization_factor)
    #print()
 
    brier = 0
    for i, count in actual.items(): # predicted probabilities
        #print(i)
        #print(proba)
        #print(normalization_factor)
        #print(probas.values())
        #print()
        for j, count2 in probas.items(): # true probabilities
            proba = count2/normalization_factor2
            #print('true: {}, {} times, predicted: {}, with probability: {}'.format(i,count,j,proba))
            if i==j:
                #print('i!: {}'.format(i))
                brier += count * math.pow(1-(proba),2)
                #print(count * math.pow(1-(proba),2))
            else:
                brier += count * math.pow(0-(proba),2)
                #print(count * math.pow(0-(proba),2))
        #print('brier for prefix {} followed by {}: {}'.format(o_prefix, i, brier))
        #print()
    #print('total brier for prefix {}: {}'.format(o_prefix, brier))
    #print()
    return brier

def brier_score(stats, normalization_factor):
    #print(normalization_factor)
    brier = 0
    #print(stats.values())
    #print(normalization_factor)
    #print()
    for i, proba in enumerate(stats.values()): # predicted probabilities
        for j, proba2 in enumerate(stats.values()): # true probabilities
            if i==j:
                brier += math.pow(1-(proba/normalization_factor),2)
            else:
                brier += math.pow(0-(proba/normalization_factor),2)
    #positive_class_brier = sum(math.pow(1-(proba/normalization_factor),2) for proba in stats.values())
    #negative_class_briers = sum(math.pow(0-(proba/normalization_factor),2) for proba in stats.values())
    #print(brier)
    return brier
    
def chomp(x):
    if x.endswith("\r\n"): return x[:-2]
    if x.endswith("\n"): return x[:-1]
    return x

all_lines = read("sepsis.txt")
random.shuffle(all_lines)
elems_per_fold = int(round(len(all_lines)/3))
train = all_lines[:2*elems_per_fold]
test = all_lines[2*elems_per_fold:]

for i in range(1,10):
    model, stats = markov_model(train, i)
    print("Order: {}, Entropy rate: {}".format(i,entropy_rate(model, stats)))
    
for i in range(1,10):
    model, stats = markov_model(chars(train), i)
    model_test, stats_test = markov_model(chars(test), i)
    print("Order: {}, Brier score: {}".format(i,total_brier_score2(model, model_test, stats_test)))
    
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

# print textwrap.fill("".join(generate(model, pick(stats), 300)))

# Copyright (C) 2013, Clement Pit--Claudel (http://pit-claudel.fr/clement/blog)
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy of 
# this software and associated documentation files (the "Software"), to deal in 
# the Software without restriction, including without limitation the rights to 
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so, 
# subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all 
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR 
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR 
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER 
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN 
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.