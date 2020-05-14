#Import necessary packages

import os
import re
from nltk.tokenize import WhitespaceTokenizer
from nltk.tokenize import sent_tokenize
from nltk.tokenize import wordpunct_tokenize
import pprint
import csv
from collections import Counter
from sklearn_crfsuite import CRF
from sklearn_crfsuite import metrics
import pickle
import nltk
import matplotlib.pyplot as plt
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_val_score , GridSearchCV
import numpy as np
import time
nltk.download('punkt')


#DRF folder reader

def get_drf(drf_path):
	drf_files = {}
	for file in os.listdir(drf_path):
		print(file)
		i = {}
		drf_list = pickle.load(open(drf_path+file,'rb'))
		i = {file : drf_list}
		drf_files.update(i)

	return drf_files

#Get sequences from the DRF files

def get_sequences(drf_list):

    temp_count = 0
    sequence = {}
    temp_dict = {}
    sentence_num = 0
    new_sequence = []

    while drf_list[temp_count][0] == 'START' and temp_count < len(drf_list):

        new_sequence = []
        new_sequence.append(drf_list[temp_count])
        temp_count += 1

        while drf_list[temp_count][0] != 'START' and temp_count < len(drf_list):

            new_sequence.append(drf_list[temp_count])
            temp_count += 1

            if temp_count == len(drf_list):
                break

        temp_dict = {sentence_num : new_sequence}
        sequence.update(temp_dict)
        sentence_num += 1

        if temp_count == len(drf_list):
            break

    return sequence

# Read the saved CRF model files stored in pickle

def get_model(file_path):
	model_files = []
	file_names = []
	for file in os.listdir(file_path):
		print(file)
		if file.endswith('.pkl'):
			model = pickle.load(open(file_path+file,'rb'))
			print("Accuracy : " +str(model.training_log_.last_iteration['item_accuracy_float']))
			print("Sentence Accuracy : " +str(model.training_log_.last_iteration['instance_accuracy_float']))
			print("F1 Score : " +str(model.training_log_.last_iteration['avg_f1']))
			print("Precision Score : " +str(model.training_log_.last_iteration['avg_precision']))
			print("Recall Score : " +str(model.training_log_.last_iteration['avg_recall']))
			file_names.append(file)
			model_files.append(model)
	return model_files,file_names


# Creating a feature dictionary to be sent in to the CRF model of the Python-CRFsuite

def feature(feature_list):
    feature_dict = {}
    temp_dict = {}
    prefix_pattern = re.compile(r'.*\*$')
    suffix_pattern = re.compile(r'^\*.*')
    string_prefix = []
    string_suffix = []

    for feature in feature_list[2]:

        if feature == "isFirstLetterCaps":
            isFirstLetterCaps = True
            feature_dict.update({"isFirstLetterCaps":True})

        elif feature == "areAllLettersCaps":
            areAllLettersCaps = True
            feature_dict.update({"areAllLettersCaps":True})

        elif feature == "containsDigit":
            containsDigit = True
            feature_dict.update({"containsDigit":True})

        elif feature == "containsSpCharacters":
            containsSpCharacters = True
            feature_dict.update({"containsSpCharacters":True})

        elif feature == "hasHyphen":
            hasHyphen = True
            feature_dict.update({"hasHyphen":True})

        elif feature == "hasApostrophe":
            hasApostrophe = True
            feature_dict.update({"hasApostrophe":True})

        elif feature == "smallLettersCapitalLetters":
            smallLettersCapitalLetters = True
            feature_dict.update({"smallLettersCapitalLetters":True})

        elif feature == "SOS":
            SOS = True
            feature_dict.update({"SOS":True})

        elif feature == "EOS" :
            EOS = True
            feature_dict.update({"EOS":True})

        elif feature == "hasDot":
            hasDot = True
            feature_dict.update({"hasDot":True})

        elif re.match(prefix_pattern,feature):
            string_prefix.append(feature)

        elif re.match(suffix_pattern,feature):
            string_suffix.append(feature)

        else:
            feature_dict.update({"word": feature})

    if len(string_prefix) > 0 :
        feature_dict.update({"string_prefix":string_prefix})
    if len(string_suffix) > 0 :
        feature_dict.update({"string_suffix":string_suffix})

    return feature_dict

# Transform the dataset of sequences into a format that will be accepted by the CRF model.

def transform_to_dataset(train_sequence):
	X , y = [] , []

	for sequence_num in range(len(train_sequence)):

	    sequence_feature = []
	    sequence_label = []

	    for word in train_sequence[sequence_num]:

	        if word[0] == '':
	            print(word)
	            continue

	        if word[1] == '':
	            print(word)
	            continue

	        if word[1] == '':
	            print(word)
	            continue

	        word_feature = feature(word)
	        sequence_feature.append(word_feature)
	        sequence_label.append(word[1])

	    X.append(sequence_feature)
	    y.append(sequence_label)

	return X , y

#plot the model evaluation such as the accuracy, sentence accuracy , loss.

def plot_model_evaluation(iterations,num , name):
	num_iterations = []
	item_accuracy_float = []
	sentence_accuracy_float = []
	loss = []
	time = []
	for iteration in iterations:
	    num_iterations.append(iteration['num'])
	    item_accuracy_float.append(iteration['item_accuracy_float'])
	    sentence_accuracy_float.append(iteration['instance_accuracy_float'])
	    loss.append(iteration['loss'])
	    time.append(iteration['time'])

	fig, axs = plt.subplots(2, 2,figsize=(15,15))
	fig.suptitle("Config {} - {} ".format(num,name),fontweight="bold", size=20,color = 'b')
	axs[0, 0].plot(num_iterations,loss)
	axs[0, 0].set_title('Loss SGD',fontweight="bold", size=15,color = 'w')
	axs[0,0].set(ylabel = 'Loss')
	axs[0, 1].plot(num_iterations,time, 'tab:orange')
	axs[0, 1].set_title('Time per iteration',fontweight="bold", size=15,color = 'w')
	axs[0,1].set(ylabel = 'Time sec')
	axs[0,1].set_yticks([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])
	axs[1, 0].plot(num_iterations,item_accuracy_float, 'tab:green')
	axs[1,0].set_yticks([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])
	axs[1, 0].set_title('Word Accuracy',fontweight="bold", size=15,color = 'w')
	axs[1,0].set(xlabel = 'Iterations' , ylabel = 'Accuracy')
	axs[1, 1].plot(num_iterations,sentence_accuracy_float, 'tab:red')
	axs[1,1].set_yticks([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])
	axs[1, 1].set_title('Sentence Accuracy',fontweight="bold", size=15 ,color = 'w')
	axs[1,1].set(xlabel = 'Iterations' , ylabel = 'Accuracy')


	plt.show()
