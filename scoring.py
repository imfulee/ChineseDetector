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

n_parameter = 1

print("Process: Train the models")
# Train the China n-gram model
china_ngram = Ngram(N=n_parameter)
china_predict = china_ngram.fit_transform(china_dataset)

# Train the Taiwan n-gram model
taiwan_ngram = Ngram(N=n_parameter)
taiwan_predict = taiwan_ngram.fit_transform(taiwan_dataset)

sentence = ''
verbose_mode = False
# Classifier
print(f"Process: Calculating Score (N={n_parameter})")
for doc in china_test:    
    sentence = str(doc[1:50]).replace('\n', '') # TODO:placed a hard limit because sometimes would multiply to 0, but must habe better way
    china_prob = china_ngram.string_prob(sentence, verbose=verbose_mode)
    taiwan_prob = taiwan_ngram.string_prob(sentence, verbose=verbose_mode)
    
    y_true.append(0) # using 0 to represent China and 1 for Taiwan
    if taiwan_prob == 0 or china_prob == 0:
        print(doc, taiwan_prob, china_prob)
    ratio = taiwan_prob / china_prob
    if ratio > 1:
        y_predicted.append(1)
    else:
        y_predicted.append(0)
        
for doc in taiwan_test:
    sentence = str(doc[1:50]).replace('\n', '')
    china_prob = china_ngram.string_prob(sentence, verbose=verbose_mode)
    taiwan_prob = taiwan_ngram.string_prob(sentence, verbose=verbose_mode)
    
    y_true.append(1)
    ratio = taiwan_prob / china_prob
    if ratio > 1:
        y_predicted.append(1)
    else:
        y_predicted.append(0)
        
print(classification_report(y_true, y_predicted, target_names=target_names))

# Moving files away from the testing database
move_files('./TestingDataset/China/', './ChinaDataset/', moved_files_number)
move_files('./TestingDataset/Taiwan/', './TaiwanDataset/', moved_files_number)
