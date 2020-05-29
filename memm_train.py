"""
Developer:          Anirban Saha. 
Version:            v0.1 (draft)
Date:               14.04.2020
Input:              Data Representation Files, Learning Rate. 
Output:             Table of weights, to predict POS tags using MEMM model.  
Description:        
Documentation Link:  
"""
import numpy as np 
import pickle
from programmes import global_ as g_  
import time
import math

"""
Description: Counts the number of occurances of ['prev_s, s, f']
Input: The DRF, list of states, list of features. 
Output: A dictionary with the count corresponding to each ['prev_s, s, f']. 
"""
def create_initial_count_dictionary(drf, dict_s, dict_f): 
  count_dict = {}  
  for line in drf: 
    word = []
    if len(line)>0:                                 # if line exists.
      prev_s = line[0].strip() 
      s = line[1].strip()                           # POS / part of speech or the label  
      key_string_1 = str(dict_s[prev_s])+","+str(dict_s[s])  
      f_set = get_active_features(line[2] , dict_f) 
      for f in f_set:
        feature_query = key_string_1 + "," + str(f)  
        if feature_query in count_dict:
            count_dict[feature_query] = count_dict[feature_query] + 1  
        else:
            count_dict[feature_query] = 1
  return count_dict

"""
Description: This is the denominator of Bishop's book's equation 4.104
Input:  Probability matrix. 
        The probability matrix has the following structure: 
        [[tag_1, weights],[tag_2, weights],...,[tag_n, weights]]
Output: Denominator equal to "Sum(exp(aj))" 
"""
def get_denominator(weight_matrix_):         #It only has the tags which has summ_of_weights>0. 
  res = 0
  maxElement =   weight_matrix_[0][1]        #log_xj is equivalent to maxElement #Johannes #statistics.java 
  for elem in weight_matrix_:
    if elem[1] > maxElement: 
        maxElement = elem[1]                 # Finds the highest of the values. 

  for class_score_pair in weight_matrix_: 
    res = res + math.exp(class_score_pair[1]  - maxElement)   # Math.exp( _vectorOfLogs[i] - maxElement ); #Johannes #statistics.java 
    
  return math.log(res) + maxElement #This is a logged value.

"""
Description:  Updates the weights in the lambda dictionary. The lambda dictionary contains the weights.
              weight = weight - learning_rate * error   
Input:  active_features, weight_dict, learning_rate, prev_s, true_tag, weight_matrix_
Output: Returns updates weight_dict 
"""
def update_weights(active_features, weight_dict, learning_rate, prev_s, true_tag, posterior_matrix_, count_dict): 

  for elem in posterior_matrix_: 
    
    tag = elem[0]
    posterior = elem[1] 
    for feature in active_features:
        search_term = str(prev_s)+","+str(tag)+","+str(feature)
        if search_term in count_dict:           
            occurances = count_dict[search_term]
        else:                                   
            occurances = 0 
        if tag == true_tag: diff = -(1 - posterior) * occurances          #multiply by the number of occurances.
        else: diff = -(0 - posterior) * occurances
        
        
        weight_dict[search_term] = weight_dict[search_term] - learning_rate * diff 
  return weight_dict

"""
Description:  Returns the numerator, required to calculate probability.
Input:  It takes in list of active features in a word, previous tag, present tag, lambda dictionary.
Output: Returns sum of weights.

Bishop's book: Equation 4.105
"""
def get_summed_weights(active_features,prev_s,i,weight_dict): #manually checked. Works!
  sum_of_weights= 0
  weight = 0 
  for feature in active_features:  
    search_term = str(prev_s)+","+str(i)+","+str(feature)  
    if search_term in weight_dict:
        weight = weight_dict[search_term] 
    else:   
        weight = 0                                  #weight=0 when s', s (this), f combination does not occur in the training corpus 
    
    sum_of_weights = sum_of_weights + weight 
  return sum_of_weights
  

"""
Description: This is called by the main function. 
"""
def train():  
  print("Welcome to Maximum Entropy Markov Model execution.")
  g_filename = input('Enter dataset name: ')
  learning_rate = input("Enter Learning Rate: ")
  stopping_criteria = input("Enter how small the difference in loss should be to terminate the training: ")
  corpus_name_validated = g_.validate_corpus_name(g_filename.strip())
  conf_num = input('Enter Configuration Number: ')
  if corpus_name_validated == 1: train_memm(g_filename, learning_rate, conf_num, stopping_criteria)
  

"""
Input:  file name of the dataset.
Output: total execution of the MEMM training.
"""
def train_memm(g_filename, learning_rate, conf_num, stopping_criteria):
  # Filename of the data-representation file (drf) 
  g_file_path = './drf/drf_'+str(conf_num)+'_'+g_filename+'_train.pkl'   
  
  # Learning Rate (from string to float) 
  learning_rate = float(learning_rate)  
  
  # Initialisation of Loss. 
  loss = -1
  prev_loss = 0
  
  # Dictionaries for the list of states and the features. 
  dict_s, dict_f = get_init_data()    #Returns the dictionaries from file system. 
    
  # List of loss values per iteration. 
  iteration_loss = []
  iteration_loss = np.asarray(iteration_loss)
  
  # File path, where the iteration loss values would be saved. 
  g_file_path_loss = "./model_params/loss_memm_"+str(conf_num)+"_"+g_filename+"_"+str(learning_rate)+"_"+str(stopping_criteria)+".csv"
  
  # This is the lower limit of the stopping criteria. 
  lower_limit_stopping_criteria = -1 * float(stopping_criteria)
  
  # Details about the final exported file.
  filename_weight_dict = "./model_params/weight_dict_"+str(conf_num)+"_"+g_filename+"_"+str(learning_rate)+"_"+str(stopping_criteria)+".pkl" 
  
  # Details about the final exported file.
  filename_count_dict = "./model_params/count_dict_"+str(conf_num)+"_"+g_filename+".pkl" 
  
  
  """
  Description: Makes a log entry of the start of the execution.  
  """
  g_.log_entry("Training started now on "+g_filename+".", g_.const_info)
  
  # Loads the Data Representation File.
  try:
    with open(g_file_path, 'rb') as file:
        global_drf_table = pickle.load(file) 
        g_.log_entry("Successfully fetched the Data Representation File.", g_.const_info)
  except:
    g_.log_entry("Failed to fetch the Data Representation File.", g_.const_error)
  
  # This is the initial lambda dictionary. (weight dictionary and lambda dictionary are the same.)
  weight_dict = create_initial_weight_dictionary(global_drf_table, dict_s, dict_f) # weight_dictp['prev_s, s, f'] = 1
  count_dict = create_initial_count_dictionary(global_drf_table, dict_s, dict_f)
  print("Weight Dictionary and Count Dictionary created.")
  no_of_tags = len(dict_s)               #Finds the number of tags. Needed in calculation of denominator of probability.
  
  #For every line in the DRF file i.e. for every word: 
  counter = 0  #counter is the number of words.
  proceed = True  #checks if the loop should run. 
  
  while (proceed == True):
    counter = counter + 1 
    for line in global_drf_table:                               #line = word = data point. 
      prev_s   = dict_s[line[0]]                                # prev_s is the POS tag of the previous word.
      true_tag = dict_s[line[1]]                                # True tag of the word. It comes from the DRF.  
      active_features = get_active_features(line[2] , dict_f)   # active_features is a list of features (numbers)   
      
      if active_features: 
        # It calculates probability value for all tags, returns the tag with maximum probability, 
        #                     the maximum probability value & the probability value of true tag. 
        # Sending active_features instead of the line[2]; replacing the string values with numeric value. It is the same. 
        log_posterior_true_tag, weight_matrix_ = get_posterior(prev_s, weight_dict, active_features, no_of_tags, true_tag)     
        if log_posterior_true_tag > 0:                                                              #Testing 
            print("log_posterior_true_tag is more than 0: "+ str(prob_true_tag)) 
        
        if len(weight_matrix_)>0:
            internal_table_lambda = update_weights(active_features, weight_dict, learning_rate, prev_s, true_tag, weight_matrix_, count_dict) 
            loss = loss - log_posterior_true_tag            #Bishop's eq. 4.108 
    
    print("Iteration: " + str(counter)+ " || Loss: " + str(loss))
    iteration_loss = np.append(iteration_loss, [loss], axis = 0)  
    if (-1*float(stopping_criteria)) <= prev_loss-loss and prev_loss-loss < float(stopping_criteria): 
      print("Process terminating")
      print(prev_loss)
      print(loss)
      proceed = False  
    else: 
      if loss < 0:
        print("The value of loss is negative. It should not be the case.") 
      if loss > prev_loss:
        print("Alert: Current Loss value "+str(loss)+"is higher than previous loss value "+str(prev_loss)+".")
        print("It should not be the case.") 
      prev_loss = loss 
      loss = 0
    np.savetxt(g_file_path_loss, iteration_loss, delimiter=",") #This is for internal use. To create graph of the loss over iteration. 

  #Export internal_table_lambda 
  f = open(filename_weight_dict, "wb")
  pickle.dump(internal_table_lambda, f) 
  f.close() 
  g_.log_entry("Model Parameters saved successfully. File path: "+filename_weight_dict, g_.const_info)
  

"""
Description:
Input:          previous pos tag, weight dictionary, active features, number of tags, true tag. 
Output:
"""
def get_posterior(prev_s, weight_dict, active_features, no_of_tags, true_tag): 
  proceed = 1  
  prob_true_tag = 0
  i = 0
  class_score_pair= []
  weight_matrix_ = [] 
  denominator = 0
  
  while i< no_of_tags: 
    score_i = get_summed_weights(active_features, prev_s,i,weight_dict)  #Equation 4.105  #Not Log value #Not exponential.
    
    class_score_pair.append(i) 
    class_score_pair.append(score_i)
    weight_matrix_.append(class_score_pair)   
        
    class_score_pair = []
    i = i +1
  
  denominator = get_denominator(weight_matrix_)         #This returns log value. 
  
  j = 0 
  while j<len(weight_matrix_) and proceed ==1:  
    try:
        log_prob = weight_matrix_[j][1] - denominator       #Bishop's eq. 4.104 
    except:
        print("Problem in Probability calculation.") 
        time.sleep(2)
    
    weight_matrix_[j][1] = log_prob 
    log_prob = 0 
    j = j+1
  
  for row in weight_matrix_:  
    if true_tag == row[0]:
      log_posterior_true_tag = row[1]  
  
  return log_posterior_true_tag, weight_matrix_  #weight_matrix_ is posterior.    


"""
Input:  Name of the dataset.
Output: Dictionary of features, states and their respective counts. 
"""
def get_init_data(): 
  try:  
    file_path_f   = './features/features.pkl'
    with open(file_path_f, 'rb') as file:
      f = pickle.load(file)  
  except: 
    g_.log_entry("Features Dictionary is not found.", g_.const_error)
    print("searched for "+file_path_f)
    return 'File not found. Find details in the log.'
  
  try:  
    file_path_s   = './tags/tags.pkl'    
    with open(file_path_s, 'rb') as file:
      s = pickle.load(file)    
  except: 
    g_.log_entry("Tags Dictionary is not found.", g_.const_error)
    print("searched for "+file_path_s)
    return 'File not found.  Find details in the log.'
  return s,f
  
"""
Description: Creates the initial weight dictionary.
Input: file path to the DRF, dictionaries of states and features.
Output: Main Table; Columns: [prev_s, s, [feature set]
"""
def create_initial_weight_dictionary(data, dict_s, dict_f): 
  weight_dict = {}
  prev_s = 0
  s = 0
  counter = 0 
  no_of_features = len(dict_f)
  no_of_tags = len(dict_s)
  total_count = no_of_tags*no_of_tags*no_of_features
  g_.log_entry("Number of features: " + str(no_of_features), g_.const_info) 
  g_.log_entry("Number of Tags: " + str(no_of_tags), g_.const_info)
  print("Creation of the Weight Dictionary starts. It takes some time. Please have patience.")
  print("Total number of entries should be "+str(total_count))
  while prev_s < no_of_tags: 
    s = 0
    while s < no_of_tags:
        f = 0
        while f < no_of_features: 
            key_string = str(prev_s)+","+str(s) + "," + str(f)
            weight_dict[key_string] = 0.0     
            counter = counter + 1 
            f = f + 1
        s = s + 1
    prev_s = prev_s + 1
  
  
  print("Total number of entries that is   "+str(counter))
  g_.log_entry("Weight dictionary created.", g_.const_info)
  return weight_dict

"""
Description: Gets the active features in an array. Returns the numbers corresponding to the active features in an array.  
"""
def get_active_features(line, dict_f):
  #line[1] is an array. 
  #dict_f is a dictionary of total features.
  feature_nums = []
  for feature in line:
    feature_nums.append(dict_f[feature]) #changedHere
  return feature_nums