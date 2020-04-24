"""
Developer:          Anirban Saha. 
Version:            v2.0 (released)
Date:               24.04.2020
Input:              Preprocessed files from Corpus Reader in Pickle format. 
Output:             Data Representation Files, Tags list, Features list with config num, test & train split, dataset names.
Description:        The file contains the feature representation of the words in a sequential format. 
Documentation Link: https://docs.google.com/document/d/1GoNc0w8WSi1TzxbzgRMrdPnGk7rg6wHxLoM-dikuuv0/edit?usp=sharing  

Version History:
Version |   Date    |  Change ID |  Changes 
1.00     03.02.2020                 Removed dead code. 
1.01     07.03.2020                 Added the word at the start of the sequence of DRF. 
1.02     15.03.2020    #Change_1    Format of the DRF changed. [prev_tag, tag, [word, features_i]]
1.03     19.04.2020                 It accepts *_test.pkl and *_train.pkl files.
1.04     24.04.2020                 Output file names contain configuration number. 
2.0      24.04.2020                 Checks whether the file already exists. If exists, it terminates.

"""
import re
import pickle
import nltk
nltk.download('punkt') 
from programmes import global_ as g_
from nltk.tokenize import word_tokenize

"""
Description: Converts a set to a dictionay. It is needed because we want to save the features, tags as a dictionary. 
Input:       Set (unique entries in a list)
Output:      Dictionary (which maps the entry with a number.)    
"""
def set_to_dict(arr_a):
  dict_a = {}
  index = 0
  for e in arr_a: 
    dict_a[e] = index
    index = index + 1
  return dict_a

"""
Description:    Checks if a word has a dot. 
Input:          word 
Output:         feature identifier
"""
def hasDot(word):
  word = word.strip()  
  if "." in word: 
    return 'hasDot'
  else:
    return ''
  
  
"""
Description:    Checks if a word has the first letter capital. 
Input:          word 
Output:         feature identifier
"""
def isFirstLetterCaps(word):
  word = word.strip() 
  try: 
    if 65 <= ord(word[0]) and ord(word[0]) <= 90: 
      # word = word.lower();
      return "isFirstLetterCaps"
    return ''
  except: return ''


"""
Description:    Checks if a word has a mix of small and capital letters. 
Input:          word 
Output:         feature identifier   
"""
def smallLettersCapitalLetters(word):
  firstLetterCaps = isFirstLetterCaps(word) 
  if firstLetterCaps == '':
    i=1
    while i< len(word):
      if 65 <= ord(word[i]) and ord(word[i]) <= 90: 
        return "smallLettersCapitalLetters"
      i = i+1
  return ''

"""
Description:    Checks if a word has all capital letters. 
Input:          word 
Output:         feature identifier   
""" 
def areAllLettersCaps(word):
  word = word.replace('.','')
  if(word):
    for letter in word: 
      if 65 > ord(letter) or ord(letter) > 90:
        return ''
  return "areAllLettersCaps"

"""
Description:    Checks if a word contains digit. 
Input:          word 
Output:         feature identifier   
"""
def containsDigit(word):
  regex = r"\w*?\d"
  hmm = re.search(regex,word)
  if hmm is not None: return "containsDigit"
  else: return ''

"""
Description:    Checks if a word contains Hyphen. 
Input:          word 
Output:         feature identifier   
"""
def hasHyphen(word):
  regex = r"-"
  hmm = re.search(regex,word)
  if hmm is not None: return "hasHyphen"
  else: return ''

"""
Description:    Checks if a word contains Apostrophe. 
Input:          word 
Output:         feature identifier   
"""
def hasApostrophe(word):
  regex = r"'"
  hmm = re.search(regex,word)
  if hmm is not None: return "hasApostrophe"
  else: return ''

"""
Description:    Checks if a word contains special characters. 
Input:          word 
Output:         feature identifier   
"""
def containsCharacters(word):
  regex = re.compile('[@_!#$%^&*()<>?/\|}{~:]')  
  if(regex.search(word) == None): return '' 
  else: return "containsSpCharacters"

"""
Description:    Finds the ngrams of characters for a word and returns them. 
Input:          word 
Output:         List containing ngrams of characters.
"""
def return_ngram_set(word, n): 
  no_of_characters = len(word)
  list_of_ngram = []
  start = 0  
  while(start <= no_of_characters - n):
    ngram = word[start:start+n]
    list_of_ngram.append('*'+ngram+'*')
    start = start + 1 
  return list_of_ngram

"""
Description:    Finds all suffixes for a word and returns them. 
Input:          word 
Output:         List containing suffixes.
"""
def return_suffix_set(word): 
  no_of_characters = len(word) * -1
  list_of_suffixes = []
  counter = -1 
  while(counter > no_of_characters):
    suffix = word[-counter:]
    list_of_suffixes.append('*'+suffix)
    counter = counter - 1 
  return list_of_suffixes

"""
Description:    Finds all prefixes for a word and returns them. 
Input:          word 
Output:         List containing prefixes.
"""
def return_prefix_set(word): 
  no_of_characters = len(word)
  list_of_prefixes = []
  counter = 1 
  while(counter < no_of_characters):
    prefix = word[0:counter]
    list_of_prefixes.append(prefix+'*')
    counter = counter + 1 
  return list_of_prefixes

"""
Description:    Takes a word, returns the features needed. 
Input:          word, word counter, length of the sentence, *params 
Output:         features of the word, unique features of the word. 
""" 
def get_encryption_for_words(word, counter, length_of_sentence, **params):  
  # Initialising the variables; default value = '' 
  prefix = '' 
  suffix = '' 
  first_letter_caps = ''
  all_caps = ''
  contains_digit = ''
  contains_sp_character = ''
  prefix_set   = []
  suffix_set   = []
  ngram_set    = []
  features = []
  
  if params.get("all_words") == 1 and word != '': 
    word_lower = word.lower()
    features.append(word_lower) 

  if params.get("flc") == 1: 
    # check for other features, like isCapitalLetter? 
    first_letter_caps = isFirstLetterCaps(word) 
    if first_letter_caps != '': features.append(first_letter_caps)  

  if params.get("alc") == 1:
    # check if all letters are capital letters.
    all_caps = areAllLettersCaps(word)
    if all_caps != '': features.append(all_caps)
  
  if params.get("contains_digit") == 1:
    # check if there is a digit inside the word.
    contains_digit = containsDigit(word)
    if contains_digit != '': features.append(contains_digit)
  
  if params.get("contains_sp_character") == 1:
    # check if there is a digit inside the word.
    contains_sp_character = containsCharacters(word)
    if contains_sp_character != '': features.append(contains_sp_character)
  
  if params.get("has_hyphen") == 1:
    # check if there is a digit inside the word.
    has_hyphen = hasHyphen(word)
    if has_hyphen != '': features.append(has_hyphen)

  if params.get("is_contraction") == 1:
    # check if there is a digit inside the word.
    has_Apostrophe = hasApostrophe(word)
    if has_Apostrophe != '': features.append(has_Apostrophe)
  
  if params.get("is_small_capital_mixed") == 1:
    is_small_cap_mixed = smallLettersCapitalLetters(word)
    if is_small_cap_mixed != '': features.append(is_small_cap_mixed)
  
  if params.get("has_dot") == 1:
    has_dot = hasDot(word)
    if has_dot != '': features.append(has_dot)
  
  word = word.lower() #makes the word to lower case. 
  
  #counter, length_of_sentence
  if (params.get("is_EOS") == 1 and counter == length_of_sentence - 2): features.append('EOS')  
  if params.get("is_SOS") ==1 and counter == 0: features.append('SOS') 

  if params.get("all_prefix") == 1: 
    prefix_set = return_prefix_set(word) 
    for prefix in prefix_set: features.append(prefix)
  
  if params.get("all_suffix") == 1: 
    suffix_set = return_suffix_set(word)
    for suffix in suffix_set: features.append(suffix)

  if params.get("all_trigram") == 1: 
    ngram_set = return_ngram_set(word, 3)
    for ngram in ngram_set: features.append(ngram)

  # Building the features set to export.
  feature_set = set()  
  for i in features: feature_set.add(i)  
  
  return features, feature_set #encoded

"""
Description:    This function takes the words (array) and tags (array) and returns the main feature representation of the word.
Input:          words (array), tags (array), *params 
Output:         List of words, Feature set.
"""
def get_list_of_words_encrypted_str(words, tags, **params): 
  #words is an array. It is the set of words in a sentence. This function will be called for every sentence. The starting word will have a previous tag of #"START".  #Change_1
  flc = 0 
  alc = 0
  contains_digit = 0
  contains_sp_character  = 0
  has_hyphen = 0
  is_contraction = 0
  is_small_capital_mixed = 0
  is_SOS = 0
  is_EOS = 0 
  all_prefix = 0
  all_suffix = 0 
  all_words  = 0  
  has_dot = 0
  prev_tag = 'START' 
  
  main_array = []
  feature_set = set()
  list_of_words = [] 
  counter = 0
  
  param_orthographic     = params.get("orthographic")
  param_morphological    = params.get("morphological")
  param_word             = params.get("word")
  
  # Values of param_* are correcting coming. 
  
  if param_orthographic == 1:   
    flc = 1 
    alc = 1
    contains_digit = 1
    contains_sp_character  = 1
    has_hyphen = 1
    is_contraction = 1
    is_small_capital_mixed = 1
    is_SOS = 1
    is_EOS = 1
    has_dot = 1
  if param_morphological == 1:
    all_prefix = 1
    all_suffix = 1
  if param_word == 1:
    all_words  = 1

  # For each line in the sentence from the Penn Tree Bank, ask for the encryption. 
  length_of_sentence = len(words)
  #print("Length of sentence:" + str(length_of_sentence)) #Length of sentence is all right. 
  while counter < len(words):
    # Entering the loop. 
    encoded_feature_array, new_features_set = get_encryption_for_words(words[counter], counter, length_of_sentence,  
                                                                       flc=flc, alc=alc, contains_digit = contains_digit, 
                                                                       contains_sp_character = contains_sp_character, 
                                                                       has_hyphen = has_hyphen, is_contraction = is_contraction, 
                                                                       is_small_capital_mixed = is_small_capital_mixed, 
                                                                       all_prefix  = all_prefix, 
                                                                       all_suffix  = all_suffix,
                                                                       is_SOS = is_SOS, 
                                                                       is_EOS = is_EOS,
                                                                       all_words = all_words, has_dot = has_dot)
        
    feature_set = feature_set.union(new_features_set)
    if not tags: 
      list_element = encoded_feature_array
      list_of_words.append(list_element)
    else:          
      main_array.append(prev_tag)                   #Change_1
      main_array.append(tags[counter])  
      main_array.append(encoded_feature_array) 
      list_of_words.append(main_array)     
      main_array = []  
      prev_tag = tags[counter]                      #Change_1
    counter = counter + 1

  return list_of_words, feature_set

"""
Description:    This function gets the words and the tags as separate arrays, from the line and returns it to the calling function.
Input:          Line
Output:         words (array), POS Tags (array). 
"""
def get_words_tags_from_line(line): 
  words = []
  posTags = []

  for word_pos_pair in line: #line is an array.
    try:
      word = word_pos_pair[0]
      pos = word_pos_pair[1]
      if word != '' and pos != '': 
        words.append(word)
        posTags.append(pos) 
    except:
      word = ''
      pos = '' 
  return words, posTags

"""
Data representation function:
- gets the feature representation of every word in the line.
- creates/updates the output file.
Input:          Line
Output:         words (array), POS Tags (array). 
"""
def data_representation(line, conf_arr): 
  #line is array format.
  words, tags = get_words_tags_from_line(line) 
  
  # Keywords: "prefix", "suffix", "flc", "alc", "params"
  # Accepted value of "params" is a string. Example: "all"
  # Accepted value of all other keywords is integer. 1 for yes. 
  
  param_word         = conf_arr[0]
  param_orthographic = conf_arr[1]
  param_morphological= conf_arr[2] 
  
  if len(words)>0: 
    main_list, feature_set = get_list_of_words_encrypted_str(words, tags, orthographic = param_orthographic, 
                                                                          morphological = param_morphological, 
                                                                          word = param_word)#params='all') 
    tag_set = set(tags) 
    return main_list, tag_set, feature_set
  else:
    return []

"""
Description:    Converts the set to a dictionary and exports them.  
Input:          set, dataset_name, choice of file. 
Output:         dictionary file in pickle format.    
"""
def export_sets(set_t, dataset_name, choice, indicator, config_num):  #indicator is to check whether it is test or train set. 
  #convert the sets to a dictionary
  dict_ = set_to_dict(set_t)

  #export them
  try: 
    if indicator == 0: filename = "./"+choice+"/"+choice+"_"+str(config_num)+"_"+dataset_name+".pkl"
    if indicator == 1: filename = "./"+choice+"/"+choice+"_"+str(config_num)+"_"+dataset_name+".pkl"
    
    f = open(filename, "wb")
    pickle.dump(dict_, f)
    f.close()
    
  except Exception as e: 
    g_.log_entry("Error dumping "+choice+" files.",g_.const_error)
    g_.log_entry(str(e), g_.const_error)
  

"""

"""
def  does_file_exist(dataset_name, config_num):
  file_path = './drf/drf_'+str(config_num)+"_"+dataset_name+'_train.pkl'
  try:
    with open(file_path, 'rb') as file:
      g_.log_entry("Data Representation Files for this dataset "+dataset_name+" exists for the configuration number "+str(config_num), g_.const_error)
      return 0
  except:
    return 1
  
  
"""
Description:    Main function, reads file, generates Data Representation Files, logs the entry. 
Input:          File path.  
Output:         Data Representation Files.
"""
def genertate_drf_files(dataset, config_num, indicator): 
  export_drf = []
  main_list = [] 
  tag_set = set()
  feature_set = set() 
  
  if indicator ==1: dataset_filepath = "./datasets/dataset_"+dataset+"_test.pkl"
  if indicator ==0: dataset_filepath = "./datasets/dataset_"+dataset+"_train.pkl"
  
  is_file_exist = does_file_exist(dataset, config_num)
  if is_file_exist == 0: 
    return 0 #failed.
  
  if is_file_exist == 1: 
    try:
      with open(dataset_filepath, 'rb') as file: 
        data = pickle.load(file)  
        num_of_rows = len(data)
        g_.log_entry("Number of rows:" + str(num_of_rows), g_.const_info)         #Number of rows not required. But it is not harming either.
        dataset_name = filepath.split('dataset_')[1].replace(".pkl",'') 
        g_.log_entry("Data representation program for "+dataset_name+" started.",g_.const_info) 
    except:
      g_.log_entry("Dataset does not exist or it does not follow naming convention.",g_.const_error)
      return g_.const_error + ' dataset does not exist.'
    
    index = 0
    conf_arr = g_.config_params[config_num] 
    while index < len(data):  
      line = data[index] #already in the form of an array. 
      if len(data[index])>0:
        try:
          main_list, set_new_tags, set_new_features = data_representation(line, conf_arr)
        except Exception as e:  
          g_.log_entry(str(e)+ ' error at ' + str(index) + ' line. Value: ' + str(data[index]), g_.const_error)
      
      if len(main_list)>0: 
        for word in main_list: export_drf.append(word) 
        tag_set = tag_set.union(set_new_tags) 
        feature_set = feature_set.union(set_new_features)
      index = index + 1 
      
    export_drp_file(export_drf, dataset_name, config_num)  #dataset_name has "_test" or "_train" in it. 
    g_.log_entry("DRP file exported successfully.", g_.const_info)
    
    tag_set.add('START')
    export_sets(tag_set, dataset_name, 'tags', indicator, config_num)
    g_.log_entry("Tags for "+dataset_name+" exported.",g_.const_info)
    export_sets(feature_set, dataset_name, 'features', indicator, config_num)
    g_.log_entry("Features for "+dataset_name+" exported.",g_.const_info)
    g_.log_entry("Data representation program for "+dataset_name+" ended.",g_.const_info) 
    return 1 #Successfully executed. 

"""
Description:    Exports the data representation files. 
Input:          list of words representations, dataset name.
Output:     
"""
def export_drp_file(main_list, dataset_name, config_num): 
  # Giving the output.
  try:
    #if indicator == 0: file_path = './drf/drf_'+str(config_num)+"_"+dataset_name+'.pkl'
    #if indicator == 1: file_path = './drf/drf_'+str(config_num)+"_"+dataset_name+'.pkl'
    file_path = './drf/drf_'+str(config_num)+"_"+dataset_name+'.pkl'
    
    file_op = open(file_path, "wb") 
    pickle.dump(main_list, file_op)
    file_op.close()

  except Exception as e: 
    g_.log_entry("Error dumping Data Representation Files.",g_.const_error)
    g_.log_entry(str(e), g_.const_error)

"""
Description:    Creates representation of words without the POS tags. 
Input:          sentence, configuration number 
Output:         feature representation of words, feature set. 
"""
def data_representation_testing(sentence, config_num): #not finalised yet.
  proceed = 1  
  try: 
    conf_arr = g_.config_params[config_num] 
    param_word         = conf_arr[0]
    param_orthographic = conf_arr[1]
    param_morphological= conf_arr[2]
  except Exception as e:
    proceed = 0 
    g_.log_entry("Configuration parameters missing.", g_.const_error)
    g_.log_entry(str(e), g_.const_error)
  
  if proceed == 1:
    words = word_tokenize(sentence)  
    empty_tags_list = [] 
    main_list, feature_list = get_list_of_words_encrypted_str(words, empty_tags_list, orthographic = param_orthographic, 
                                                                                      morphological = param_morphological,
                                                                                      word = param_word)
    return main_list, feature_list