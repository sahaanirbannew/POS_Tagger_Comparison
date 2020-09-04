import pickle
import os
import pprint
import time

"""
The Out of Vocabulary (OOV) --> The words that are not present in the training dataset
These words are extracted from the test data set and compiled in a dictionary in the following format.

{ OOV_word : [Actual_tag_list (ground_truth) of the respective word]}

Example dictionary:
{
 'radical': ['ADJ'],
 'rage': ['ADJ', 'NOUN', 'VERB', 'PROPN'],
 'raid': ['ADJ', 'NOUN', 'VERB'],
 'raiding': ['VERB'],
 'rambled': ['VERB'],
 'ranchers': ['NOUN'],
 'ranking': ['NOUN', 'ADJ']
 }

Using this dictionary Unknown Word Accuracy or OOV word Accuracy can be calculated.
"""


def create_token_list(data_dictionary):

    tokens_list = []
    for sent_num, tokens in data_dictionary.items():
        for token in tokens:
            tokens_list.append(token[0])

    return tokens_list


def find_oov_words(train_data_path, test_data_path):
    train_data = pickle.load(open(train_data_path, "rb"))
    test_data = pickle.load(open(test_data_path, "rb"))

    train_set_tokens_list = create_token_list(train_data)
    test_set_tokens_list = create_token_list(test_data)

    # print(len(train_set_tokens_list), len(test_set_tokens_list))

    train_set_vocabulary = set(train_set_tokens_list)
    test_set_vocabulary = set(test_set_tokens_list)

    # print(len(train_set_vocabulary), len(test_set_vocabulary))

    # Finding the tokens only present in test data set. test_data.difference(train_data)
    oov_word_set = test_set_vocabulary.difference(train_set_vocabulary)
    # print(len(oov_word_set))

    oov_percent = (len(oov_word_set) / len(test_set_vocabulary)) * 100
    # print(oov_percent)

    return oov_word_set, train_set_vocabulary, test_set_vocabulary


def find_oov_word_tags(oov_word_set, test_data_path):
    oov_word_tag_dict = {}

    test_data = pickle.load(open(test_data_path, "rb"))

    for oov_word in oov_word_set:
        tag_list = []
        for key, tokens in test_data.items():
            for token in tokens:
                if oov_word in token[0]:
                    if token[1] in tag_list:
                        continue
                    tag_list.append(token[1])
        oov_word_tag_dict.update({oov_word: tag_list})

    return oov_word_tag_dict


penn_train_data_path = "datasets/dataset_penn_train.pkl"
penn_test_data_path = "datasets/dataset_penn_test.pkl"

genia_train_data_path = "datasets/dataset_genia_train.pkl"
genia_test_data_path = "datasets/dataset_genia_test.pkl"

conll_train_data_path = "datasets/dataset_conll_train.pkl"
conll_test_data_path = "datasets/dataset_conll_test.pkl"

# Same Domain

# Trained : Penn        Tested : Penn

penn_test_oov_words, penn_train_vocabulary, penn_test_vocabulary = find_oov_words(
    penn_train_data_path, penn_test_data_path
)

penn_oov_word_tag_dict = find_oov_word_tags(penn_test_oov_words, penn_test_data_path)

pickle.dump(penn_oov_word_tag_dict, open("oov_words/penn_on_penn_oov_dict.pkl", "wb"))


# Trained : Genia           Tested : Genia

genia_test_oov_words, genia_train_vocabulary, genia_test_vocabulary = find_oov_words(
    genia_train_data_path, genia_test_data_path
)

genia_oov_word_tag_dict = find_oov_word_tags(genia_test_oov_words, genia_test_data_path)

pickle.dump(
    genia_oov_word_tag_dict, open("oov_words/genia_on_genia_oov_dict.pkl", "wb")
)


# Trained : CoNLL           Tested : CoNLL

conll_test_oov_words, conll_train_vocabulary, conll_test_vocabulary = find_oov_words(
    conll_train_data_path, conll_test_data_path
)

conll_oov_word_tag_dict = find_oov_word_tags(conll_test_oov_words, conll_test_data_path)

pickle.dump(
    conll_oov_word_tag_dict, open("oov_words/conll_on_conll_oov_dict.pkl", "wb")
)


# Cross Domain

# Trained : Penn     Tested : Genia

penn_genia_test_oov_words, _, _ = find_oov_words(
    penn_train_data_path, genia_test_data_path
)

penn_genia_oov_word_tag_dict = find_oov_word_tags(
    penn_genia_test_oov_words, genia_test_data_path
)

pickle.dump(
    penn_genia_oov_word_tag_dict, open("oov_words/penn_on_genia_oov_dict.pkl", "wb")
)

# Trained : Penn     Tested : CoNLL

penn_conll_test_oov_words, _, _ = find_oov_words(
    penn_train_data_path, conll_test_data_path
)

penn_conll_oov_word_tag_dict = find_oov_word_tags(
    penn_conll_test_oov_words, conll_test_data_path
)

pickle.dump(
    penn_conll_oov_word_tag_dict, open("oov_words/penn_on_conll_oov_dict.pkl", "wb")
)

# Trained : Genia    Tested : Penn

genia_penn_test_oov_words, _, _ = find_oov_words(
    genia_train_data_path, penn_test_data_path
)

genia_penn_oov_word_tag_dict = find_oov_word_tags(
    genia_penn_test_oov_words, penn_test_data_path
)

pickle.dump(
    genia_penn_oov_word_tag_dict, open("oov_words/genia_on_penn_oov_dict.pkl", "wb")
)

# Trained : Genia    Tested : CoNLL

genia_conll_test_oov_words, _, _ = find_oov_words(
    genia_train_data_path, conll_test_data_path
)

genia_conll_oov_word_tag_dict = find_oov_word_tags(
    genia_conll_test_oov_words, conll_test_data_path
)

pickle.dump(
    genia_conll_oov_word_tag_dict, open("oov_words/genia_on_conll_oov_dict.pkl", "wb")
)

# Trained : CoNLL    Tested : Penn

conll_penn_test_oov_words, _, _ = find_oov_words(
    conll_train_data_path, penn_test_data_path
)

conll_penn_oov_word_tag_dict = find_oov_word_tags(
    conll_penn_test_oov_words, penn_test_data_path
)

pickle.dump(
    conll_penn_oov_word_tag_dict, open("oov_words/conll_on_penn_oov_dict.pkl", "wb")
)

# Trained : CoNLL    Tested : Genia

conll_genia_test_oov_words, _, _ = find_oov_words(
    conll_train_data_path, genia_test_data_path
)

conll_genia_oov_word_tag_dict = find_oov_word_tags(
    conll_genia_test_oov_words, genia_test_data_path
)

pickle.dump(
    conll_genia_oov_word_tag_dict, open("oov_words/conll_on_genia_oov_dict.pkl", "wb")
)

