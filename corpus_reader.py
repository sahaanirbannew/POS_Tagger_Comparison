# -*- coding: utf-8 -*-
"""

Developer         :      Venkatesh Murugadas
Version           :      0.04
Date              :      05/04/2020
Input             :      None
Output            :      None
Description       :      This file defines the functions for parsing the pentree bank, Genia and CoNLL corpus used for Parts of Speech Tagging.
Documentation Link:      https://docs.google.com/document/d/16GPkM6cVHHhpFTJk_WOIH4i23Ap9GiHmSvzJfmZYHRE/edit?usp=sharing

Version History   :
- 0.02
- 02/03/2020
- Change : 1. The ambiguous tags are converted into single tags
           2. The Pan/E2A token is ignored since it is wrongly tokenised
- Change ID : #AmbiguousTagCon

- 0.03
- 25/03/2020
- Change : 1. Adding the function to create test and train split of the corpus.
- Change ID : #TestTrainSplit

"""


"""

Utilities

Load essential Libraries

Import all the necessary packages that is necessary for the reading of the dataset

"""


import os  # loads the files in the directory
import errno  # error number
import re  # regular expression package
import nltk  # Natural language tool kit
from nltk.tokenize import WhitespaceTokenizer  # word tokenisation
from nltk.tokenize import sent_tokenize  # sentence tokenisation
import pprint  # print the dictionary in a legible format
import csv  # read and write csv files
from collections import Counter  # used as a counter
import pickle  # serialising the data

nltk.download("punkt")  # for tokenisation
import itertools  # for slicing the dictionary


def get_data(file_path):

    """

    This function is used to load the corpus files

    To get the files present in the directory mentioned above.

    Make sure there are only set of Tagged files present in the folder given in the file path. Remove the unncessary files.

    param  : File path of the corpus
    return : The corpus file(s) present in the folder


    """

    # list of all the files
    files = []
    for file in os.listdir(file_path):
        # The files end with .pos
        if file.endswith(".pos"):
            f = open(file_path + file, "r", encoding="utf-8")
            files.append(f.read())
        # The files end with .txt
        elif file.endswith(".txt"):
            f = open(file_path + file, "r", encoding="utf-8")
            files.append(f.read())

    return files


"""

Functions to Create Dataset

"""


def initialise_variables():
    """

    This function is used to initialise the variables used in the dataset creation

    praram : None

    return :
            file_list
                - list of all the file names

             tagged_token
                - temporary list of words and respective tags used to update the dictionary
                  [['word1','tag1']['word2','tag2']] - single sentence

             total_tagged_tokens
                - a total list of words and their respective tags in the corpus
                  [[['word1','tag1']['word2','tag2']][['word1','tag1']['word2','tag2']]] - all sentences in the corpus

             dictionary
                - The main dictionary with the sentence number and the sentence with words and tags.

                  {0 : [['word1','tag1'],['word2','tag2']],
                   1 : [['word3','tag3'],['word4','tag4']],
                   .
                   .
                   .
                   N : [['wordN-1','tagN-1'],['wordN','tagN']] }

             i
                - the variable used to assign the sentence number
             d
                - the temporary dictionary used to update the main dictionary
                  { i  : [['word1','tag1'],['word2','tag2']]}
    """

    file_list = []
    tagged_token = []
    total_tagged_tokens = []
    dictionary = {}
    i = 0
    d = {}

    return file_list, tagged_token, total_tagged_tokens, dictionary, i, d


def create_penntreebank_dataset(files):

    """
     This function is used to create Penn Treebank dataset from the tagged corpus.

     Example : Annotation of the corpus

     ======================================

     [ Heritage/NNP Media/NNP Corp./NNP ]
     ,/,
     [ New/NNP York/NNP ]
     ,/, said/VBD
     [ it/PRP ]
     offered/VBD to/TO buy/VB
     [ the/DT shares/NNS ]
     of/IN
     [ POP/NNP Radio/NNP Corp./NNP it/PRP ]
     does/VBZ n't/RB already/RB own/VB in/IN
     [ a/DT stock/NN swap/NN ]
     ./.

     ======================================


     param :  files
              The files read from the corpus folder with the get_data() function

     return : dictionary
              The main dictionary with the sentence number and the sentence with words and tags.

              {0 : ['START',['word1','tag1'],['word2','tag2']],
               1 : ['START',['word3','tag3'],['word4','tag4']],
               .
               .
               .
               N : ['START',['wordN-1','tagN-1'],['wordN','tagN']] }

     """
    # initalise the variables for creating the dataset
    (
        file_list,
        tagged_token,
        total_tagged_tokens,
        dictionary,
        i,
        d,
    ) = initialise_variables()

    # pattern to recognise the words
    pattern_word = re.compile(r"^.*\D\\\/\D.*$")

    # pattern to recognise the digits
    pattern_digit = re.compile(r"^.*\d\\?\\\/\d.*$")

    for file in files:
        # replace the '[]' with space and read all the text present within a file
        text = re.sub(r"(\[)|(\])", r" ", file, flags=re.M)
        # tokenise the texts into sentences using NLTK
        sents = sent_tokenize(text)
        for sent in sents:
            # the list to store the words and tags after removing the unnecessary tokens
            tokens_modified = []
            tagged_token = []

            # Tokenise the sentences into words using NLTK
            tokens = WhitespaceTokenizer().tokenize(sent)

            # Removing the unnecessary tokens present in the corpus
            for token in tokens:
                # these tokens are used to separate the sentences in the tagged corpus
                if token == "======================================":
                    continue
                else:
                    tokens_modified.append(token)
                # check if the created list is empty or not
                if len(tokens_modified) == 0:
                    continue

            """

            From the list of modified tokens , search for patterns with word1\/word2
            and replace "\/" with space, num1\/num2 and replace "\/" with no_space.

            Example:

            Property\/casualty/NN
            1\/2/CD

            """
            for token in tokens_modified:

                if re.match(pattern_word, token) is not None:
                    token = re.sub(r"\\?\\\/", r" ", token)
                elif re.match(pattern_digit, token) is not None:
                    token = re.sub(r"\\?\\\/", r"", token)
                else:
                    # tagged_token is the final list with [word,tag] for the particular sentence.
                    tagged_token.append(token.split("/"))

            # condition to check that there are no empty lines added to the dictionary
            if len(tagged_token) != 0:
                # temporary dictionary that is used to update the main dictionary
                d = {i: tagged_token}
            # main dictionary with { 'sentence_line' : sentence } format
            dictionary.update(d)
            # main list with all the sentences
            total_tagged_tokens.append(tagged_token)
            # increment the sentence line
            i = i + 1

        # list of all raw text present in the files
        file_list.append(text)

    return dictionary


def create_genia_dataset(files):
    """

      This function is used to create Genia dataset from the tagged corpus.

      Example : Annotation of the Genia corpus

      activation/NN
      resulting/VBG
      in/IN
      enhanced/VBN
      production/NN
      of/IN
      interleukin-2/NN
      (/(
      IL-2/NN
      )/)
      and/CC
      cell/NN
      proliferation/NN
      ./.
      ====================
      In/IN
      primary/JJ
      T/NN


      param :  files
               The files read from the corpus folder with the get_data() function

      return : dictionary
               The main dictionary with the sentence number and the sentence with words and tags.

               {0 : [['word1','tag1'],['word2','tag2']],
                1 : [['word3','tag3'],['word4','tag4']],
                .
                .
                .
                N : [['wordN-1','tagN-1'],['wordN','tagN']] }

    """
    # initalise the variables for creating the dataset
    (
        file_list,
        tagged_token,
        total_tagged_tokens,
        sentence_dict,
        i,
        d,
    ) = initialise_variables()

    # pattern to recognise the string word/tag format
    patt = re.compile(".*\/.*")

    for file in files:
        # replace the '[]' with space and read all the text present within a file
        text = re.sub(r"(\[)|(\])", r" ", file, flags=re.M)
        # tokenise the texts into sentences using NLTK
        sents = sent_tokenize(text)
        for sent in sents:
            new_tagged_token = []
            tagged_token = []

            # the list to store the words and tags after removing the unnecessary tokens
            tokens_modified = []
            # Tokenise the sentences into words using NLTK
            tokens = WhitespaceTokenizer().tokenize(
                sent
            )  # Example : ['====================', 'Copyright/NN', '1999/CD', 'Academic/NNP', 'Press/NNP', './.']

            for token in tokens:

                # these tokens are used to separate the lines in the tagged corpus
                if token == "====================":
                    continue

                elif token == ".====================":
                    continue

                else:
                    tokens_modified.append(
                        token
                    )  # Example : ['To/TO', 'date/VB', 'no/DT', 'SOCS/NN', 'proteins/NNS', 'have/VBP']

                # check if the created list is empty or not
            if len(tokens_modified) == 0:
                continue

            # the modified list is checked whether it satifies this pattern of word/tag format and it is append to a list for further processing.
            for token in tokens_modified:
                if patt.match(token):

                    tagged_token.append(
                        token.split()
                    )  # Example : [['This/DT'], ['suppressive/JJ'], ['effect/NN'], ['was/VBD']]

            """

            Example : [['This/DT'], ['suppressive/JJ'], ['effect/NN'], ['was/VBD']]

            The word and the tag is separated by "/"

            The characters on the left side of the '/' are words and to the right are tags.

            In all the cases the tags are present at the right most end after the '/'.
                Example :  NF-kappaB/Rel/NN , Platelet/endothelium/NN etc.

            So from the above example NF-kappaB/Rel is split separatedly and joined with a space in between
            such as 'NF-kappaB Rel'. 'NN' is split separately as the tag.

            """
            # print(len(tagged_token))
            for token in tagged_token:
                for t in token:
                    # This instance is not correctly tokenized , so we can ignore this. There are other instances of 'Pan/E2A/NN'. So this can be ignored.
                    if t == "Pan/E2A":  # AmbiguousTagCon
                        continue
                    # temporary list to append the word and tag
                    list_1 = []
                    word_lis = []
                    tag = []
                    word_lis = t.split("/")[
                        0:-1
                    ]  # Example : Platelet/endothelium/NN --> Platelet/endothelium --> Platelet endothelium
                    word = " ".join(word_lis)

                    tag = t.split("/")[-1]  # Example : Platelet/endothelium/NN --> NN

                    # checking for empty word list
                    if word == "":
                        continue
                    # checking for empty tags or '' or : --> they have no characters to the left
                    if tag == "" or tag == "''" or tag == ":":
                        continue

                    list_1.append(word)
                    list_1.append(tag)
                # check for empty list
                if len(list_1) > 0:
                    new_tagged_token.append(list_1)

            # condition to check that there are no empty lines added to the dictionary
            if len(new_tagged_token) != 0:
                # temporary dictionary that is used to update the main dictionary
                d = {i: new_tagged_token}
            # main dictionary with { 'sentence_line' : sentence } format
            sentence_dict.update(d)
            # main list with all the sentences
            total_tagged_tokens.append(tagged_token)
            # increment the sentence line
            i = i + 1

        # list of all raw text present in the files
        file_list.append(text)

    return sentence_dict


def create_conll_dataset(files):
    """

      This function is used to create CoNLL dataset from the tagged corpus.

      Example : Annotation of the CoNLL corpus

      The first column contains the current word, the
      second its part-of-speech tag as derived by the Brill tagger and the
      third its chunk tag as derived from the WSJ corpus.

      August NNP I-NP
      's POS B-NP
      near-record JJ I-NP
      deficits NNS I-NP
      . . O

      Chancellor NNP O
      of IN B-PP
      the DT B-NP

      param  :  files
               The files read from the corpus folder with the get_data() function

      return : dictionary
               The main dictionary with the sentence number and the sentence with words and tags.

               {0 : [['word1','tag1'],['word2','tag2']],
                1 : [['word3','tag3'],['word4','tag4']],
                .
                .
                .
                N : [['wordN-1','tagN-1'],['wordN','tagN']] }

    """
    # initalise the variables for creating the dataset
    (
        file_list,
        tagged_token,
        total_tagged_tokens,
        sentence_dict,
        i,
        d,
    ) = initialise_variables()  # initalise the variables for creating the dataset

    # pattern to recognise the end of the sentence in the conll tagged corpus
    pattern_n = re.compile("\n\s")

    for file in files:
        # replace the '[]' with space and read all the text present within a file
        text = re.sub(r"(\[)|(\])", r" ", file, flags=re.M)
        # tokenise the texts into sentences using NLTK
        sents = nltk.regexp_tokenize(text, pattern_n, gaps=True)

        for sent in sents:
            tagged_token = []

            # Tokenise the sentences into words by splitting it with '\n'
            tokens = sent.split("\n")

            for token in tokens:
                """
                Example : August NNP I-NP

                August --> Word
                NNP --> POS tag
                I-NP --> Chunk tag

                """
                token_n = token.split(" ")[0:-1]  # August NNP I-NP --> ['August','NNP']
                # check for empty or single element in token_n list
                if len(token_n) == 0:
                    continue
                if len(token_n) == 1:
                    continue

                tagged_token.append(token_n)
            # condition to check that there are no empty lines added to the dictionary
            if len(tagged_token) != 0:
                # temporary dictionary that is used to update the main dictionary
                d = {i: tagged_token}

            # main dictionary with { 'sentence_line' : sentence } format
            sentence_dict.update(d)
            # main list with all the sentences
            total_tagged_tokens.append(tagged_token)
            # increment the sentence line
            i = i + 1
        # list of all raw text present in the files
        file_list.append(text)

    return sentence_dict


"""

Modify tags to Universal Tags

The three corpus consists of various tagset.

The tagsets are normalised to universal tagset for cross-domain training and testing.

Conversion of Penn Tree bank tagsets to Universal Tagsets --> https://universaldependencies.org/tagset-conversion/en-penn-uposf.html

Universal POS tags
*https://universaldependencies.org/u/pos/index.html*

ADJ: adjective
ADP: adposition
ADV: adverb
AUX: auxiliary
CCONJ: coordinating conjunction
DET: determiner
INTJ: interjection
NOUN: noun
NUM: numeral
PART: particle
PRON: pronoun
PROPN: proper noun
PUNCT: punctuation
SCONJ: subordinating conjunction
SYM: symbol
VERB: verb
X: other

"""


def modify_tags(dictionary):  # AmbiguousTagCon
    """
    This function is used to normalise the Penn Treebank, Genia and CoNLL tagset into Universal tagset

    For abbreviation of the tagset please look into the documentation.

    param : dictionary
            The main dictionary with the sentence number and the sentence with words and tags.

             {0 : [['word1','tag1'],['word2','tag2']],
              1 : [['word3','tag3'],['word4','tag4']],
              .
              .
              .
              N : [['wordN-1','tagN-1'],['wordN','tagN']] }

    return : dictionary
             The modified main dictionary with the same format as above.


    """
    for item in range(len(dictionary)):
        for tags in dictionary[item]:
            if tags[1] == "#" or tags[1] == "$" or tags[1] == "SYM":
                # Symbols
                tags[1] = "SYM"
            elif (
                tags[1] == ","
                or tags[1] == "."
                or tags[1] == ":"
                or tags[1] == "("
                or tags[1] == ")"
                or tags[1] == "``"
                or tags[1] == "''"
                or tags[1] == "-"
            ):
                # Punctuation
                tags[1] = "PUNCT"
            elif (
                tags[1] == "AFX"
                or tags[1] == "JJ"
                or tags[1] == "JJR"
                or tags[1] == "JJS"
                or tags[1] == "JJ|NN"
                or tags[1] == "JJ|NNS"
                or tags[1] == "JJ|VBN"
                or tags[1] == "JJ|RB"
                or tags[1] == "JJ|VBG"
            ):
                # Adjective
                tags[1] = "ADJ"
            elif tags[1] == "CC":
                # Conjugation
                tags[1] = "CCONJ"
            elif tags[1] == "CD":
                # Numbers
                tags[1] = "NUM"
            elif (
                tags[1] == "DT"
                or tags[1] == "PDT"
                or tags[1] == "PRP$"
                or tags[1] == "WDT"
                or tags[1] == "WP$"
                or tags[1] == "CT"
                or tags[1] == "XT"
            ):
                # Articles or Determinants
                tags[1] = "DET"
            elif (
                tags[1] == "EX"
                or tags[1] == "PRP"
                or tags[1] == "WP"
                or tags[1] == "PP"
            ):
                # Pronoun
                tags[1] = "PRON"
            elif tags[1] == "FW" or tags[1] == "LS" or tags[1] == "NIL":
                # Foreign term
                tags[1] = "X"
            elif (
                tags[1] == "IN"
                or tags[1] == "RP"
                or tags[1] == "IN|PRP$"
                or tags[1] == "IN|CC"
            ):
                # Adposition
                tags[1] = "ADP"
            elif (
                tags[1] == "MD"
                or tags[1] == "VB"
                or tags[1] == "VBD"
                or tags[1] == "VBG"
                or tags[1] == "VBN"
                or tags[1] == "VBP"
                or tags[1] == "VBZ"
                or tags[1] == "VBG|NN"
                or tags[1] == "VBP|VBG"
                or tags[1] == "VBG|JJ"
                or tags[1] == "VBD|VBN"
                or tags[1] == "VBN|JJ"
                or tags[1] == "VBP|VBZ"
            ):
                # Verb
                tags[1] = "VERB"
            elif (
                tags[1] == "NN"
                or tags[1] == "NNS"
                or tags[1] == "NN|NNS"
                or tags[1] == "NN|CD"
                or tags[1] == "NNS|FW"
                or tags[1] == "NN|DT"
                or tags[1] == "N"
            ):
                # Noun
                tags[1] = "NOUN"
            elif tags[1] == "NNP" or tags[1] == "NNPS":
                # Proper noun
                tags[1] = "PROPN"
            elif tags[1] == "POS" or tags[1] == "TO":
                # Particle
                tags[1] = "PART"
            elif (
                tags[1] == "RBR"
                or tags[1] == "RBS"
                or tags[1] == "RB"
                or tags[1] == "WRB"
            ):
                # Adverb
                tags[1] = "ADV"
            elif tags[1] == "UH":
                # Interjection
                tags[1] = "ITNJ"

    return dictionary


"""

Statistical Data collection of the corpus

"""


def create_word_list(dictionary):
    """
    This function is used to create a list of all the words and word counter.

    param : dictionary
            The main dictionary with the sentence number and the sentence with words and tags.

             {0 : [['word1','tag1'],['word2','tag2']],
              1 : [['word3','tag3'],['word4','tag4']],
              .
              .
              .
              N : [['wordN-1','tagN-1'],['wordN','tagN']] }

    return : word_list
             The list of all words present in the main dictionary

             word_counter
             A dictionary with unique words as key and their count as values

    """
    word_list = []
    word_counter = {}
    for item in range(len(dictionary)):
        for tags in dictionary[item]:
            # list of words
            word_list.append(tags[0])
    # Count of the words
    word_counter = Counter(word_list)
    return word_list, word_counter


def create_tag_list(dictionary):
    """
    This function is used to create a list of all the tags and tag counter.

    param : dictionary
            The main dictionary with the sentence number and the sentence with words and tags.

             {0 : [['word1','tag1'],['word2','tag2']],
              1 : [['word3','tag3'],['word4','tag4']],
              .
              .
              .
              N : [['wordN-1','tagN-1'],['wordN','tagN']] }

    return : tag_list
             The list of all tags present in the main dictionary

             tag_counter
             A dictionary with unique tags as key and their count as values

    """
    tag_list = []
    tag_counter = {}
    for item in range(len(dictionary)):
        for tags in dictionary[item]:
            # list of tags
            tag_list.append(tags[1])
    # Count of Tags
    tag_counter = Counter(tag_list)
    return tag_list, tag_counter


def create_sentences_list(dictionary):
    """
    This function is used to create a list of all the sentences .

    param : dictionary
            The main dictionary with the sentence number and the sentence with words and tags.

             {0 : [['word1','tag1'],['word2','tag2']],
              1 : [['word3','tag3'],['word4','tag4']],
              .
              .
              .
              N : [['wordN-1','tagN-1'],['wordN','tagN']] }

    return : setence_list
             The list of all sentences present in the main dictionary

    """
    sentences_list = []
    for item in range(len(dictionary)):
        # list of sentences
        sentences_list.append(dictionary[item])
    return sentences_list


"""

Outpul File Writing and Reading

"""


def write_pickle(file_name, dictionary):
    """
    This function is used to write the main dictionary into a pickle file.

    param : file_name
            The file name of the pickle files

            dictionary
            The main dictionary with the sentence number and the sentence with words and tags.

             {0 : [['word1','tag1'],['word2','tag2']],
              1 : [['word3','tag3'],['word4','tag4']],
              .
              .
              .
              N : [['wordN-1','tagN-1'],['wordN','tagN']] }

    The output of this function is pickle file

    """
    with open(file_name, "wb") as pickle_file:
        pickle_file.write(pickle.dumps(dictionary))


def read_pickle(file_path):
    """
    This function is used to read the main dictionary into a pickle file.

    param : file_path
            The file path to the pickle file

    return : data_dictionary
             The dictionary stored in the pickle file.

    """

    with open(file_path, "rb") as file:
        data_dictionary = pickle.load(file)

    return data_dictionary


def write_txt(file_name, sentences_list):
    """
    This function is used to write the sentences of the corpus as text file.

    param : file_name
            The file name of the text file.

            sentences_list
            The list of sentences to be written in the text file.

    The output of this function is text file

    """
    j = 0
    with open(file_name, "w") as f:
        for item in sentences_list:
            f.write("{0} : {1} \n".format(j, item))
            j = j + 1


def write_csv(file_name, dictionary):
    """
    This function is used to write the dictionary as csv - Comma separated values file.

    param : file_name
            The file name of the csv file.

            dictionary
            The main dictionary with the sentence number and the sentence with words and tags.


    The output of this function is csv file

    """
    with open(file_name + ".csv", "w") as csv_file:
        writer = csv.writer(csv_file)
        for key, value in dictionary.items():
            writer.writerow([key, value])


def write_tsv(file_name, dictionary):

    """
    This function is used to write the dictionary as tsv - Tab Separated Values file.

    param : file_name
            The file name of the tsv file.

            dictionary
            The main dictionary with the sentence number and the sentence with words and tags.


    The output of this function is tsv file

    """
    with open(file_name + ".tsv", "w") as csv_file:
        writer = csv.writer(csv_file)
        for key, value in dictionary.items():
            writer.writerow([key, value])


def read_corpus():

    """
    This function is used to initate the program by asking the user to choose an option for reading the corpus.

    param : None

    return : option
             The option for reading of the three corpora.
    """

    print("Enter the option for reading a corpus")
    print("To exit the program , please enter 'exit' ")
    print("Option 1 : Penn Tree bank corpus")
    print("Option 2 : Genia corpus")
    print("Option 3 : CoNLL corpus")

    option = input("Please choose your option : ")

    return option


def test_train_split(modified_dictionary):  # TestTrainSplit
    """
    This function is used to create the test train split of the modified corpus which stored in the format of a dictionary with
    sentence index as key and the tagged sentence as the value.

    param : modified_dictionary
            The dictionary that contains the sentence number as key and the tagged sentence as values. The tags are normalised in this dictionary.

    return : train_dictionary
             test_dictionary

             The modified_dictionary is split into two dictionary.

    """

    total_sentences = len(modified_dictionary)

    train_percentage = int(input("Enter the Percentage of train data : "))

    train_data = int((train_percentage / 100) * total_sentences)
    train_dictionary = dict(
        itertools.islice(modified_dictionary.items(), 0, train_data + 1)
    )

    test_data = int(((train_percentage - 100) / 100) * total_sentences)
    test_dictionary = dict(
        itertools.islice(modified_dictionary.items(), train_data + 1, total_sentences)
    )

    print("The number of train data")
    print(len(train_dictionary))

    print("The number of test data")
    print(len(test_dictionary))

    return train_dictionary, test_dictionary
