from nltk.util import *
from nltk.lm.preprocessing import *
from sklearn.metrics import classification_report
import os
from ngram import *
from preprocessor import *

y_true = []
y_predicted = []
target_names = ['China', 'Taiwan']

moved_files_number = 3000
# Moving files into the testing dataset     
move_files('./ChinaDataset/', './TestingDataset/China/', moved_files_number)
move_files('./TaiwanDataset/', './TestingDataset/Taiwan/', moved_files_number)

# Read the training files
print("Process: Read the files")
china_dataset = files_to_list('./ChinaDataset/')
taiwan_dataset = files_to_list('./TaiwanDataset/')
china_test = files_to_list('./TestingDataset/China/')
taiwan_test = files_to_list('./TestingDataset/Taiwan/')

# Change all the strings in list to be of list type
china_dataset = [list(doc) for doc in china_dataset]
taiwan_dataset = [list(doc) for doc in taiwan_dataset]

N = 3

print(f"Process: Building n-grams with N={N}")
# Train the China n-gram model
china_train, china_vocab = padded_everygram_pipeline(order=N, text=china_dataset)
taiwan_train, taiwan_vocab = padded_everygram_pipeline(order=N, text=taiwan_dataset)

print("Process: Train the model")
from nltk.lm import Lidstone
gamma_param = 0.5
china_model = Lidstone(gamma=gamma_param, order=N)
china_model.fit(china_train, china_vocab)
taiwan_model = Lidstone(gamma=gamma_param, order=N)
taiwan_model.fit(taiwan_train, taiwan_vocab)

import math

def log_score(model, N, sentence):
    log_score = 0.0
    sentence = pad_both_ends(list(sentence), n=N)
    ngram_sents = list(ngrams(sentence, n=N))
    for ngram_sent in ngram_sents:
        log_score += math.log(model.unmasked_score(word=ngram_sent[-1], context=ngram_sent[0:-1]))
    return log_score


sentence = ''
# Classifier
print(f"Process: Calculating F1 Score (N={N})")
for index, doc in enumerate(china_test):
    print(f"China -> {index}", end='\r')
    china_prob = log_score(china_model, N, doc)
    taiwan_prob = log_score(taiwan_model, N, doc)
    
    y_true.append(0) # using 0 to represent China and 1 for Taiwan
    if taiwan_prob > china_prob:
        y_predicted.append(1)
    else:
        y_predicted.append(0)

print("China -> Done")

for index, doc in enumerate(taiwan_test):
    print(f"Taiwan -> {index}", end='\r')
    china_prob = log_score(china_model, N, doc)
    taiwan_prob = log_score(taiwan_model, N, doc)

    y_true.append(1)
    if taiwan_prob > china_prob:
        y_predicted.append(1)
    else:
        y_predicted.append(0)

print("Taiwan -> Done")

print(classification_report(y_true, y_predicted, target_names=target_names))

# Moving files away from the testing database
move_files('./TestingDataset/China/', './ChinaDataset/', moved_files_number)
move_files('./TestingDataset/Taiwan/', './TaiwanDataset/', moved_files_number)