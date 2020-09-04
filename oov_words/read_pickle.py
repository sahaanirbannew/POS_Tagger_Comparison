"""
This is just a test program to read the pickle files
"""

import pickle
import pprint

file_path = "oov_words/penn_on_genia_oov_dict.pkl"

conll_oov_word_dict = pickle.load(open(file_path, "rb"))

pprint.pprint(conll_oov_word_dict)

