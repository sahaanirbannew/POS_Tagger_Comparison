from programmes import data_representation, global_

"""
List of valid config numbers and their interpretations: 
Config_num      Word        Orthographic Features     Morphological Features 
0               yes         no                        yes   
1               yes         no                        no
2               yes         yes                       yes                   << all features.
3               yes         yes                       no
4               no          no                        yes 
5               no          yes                       yes
6               no          yes                       no
"""

def test(): 
  characters= ['.','!', '?']
  
  """
  A little bit of pre-processing is needed. 
  """
  sentence = input("Enter a sentence: ")
  if sentence[-1] not in characters: sentence = sentence + '.'
  
  """
  Description: Calls the data representation program.
  Input:       sentence, configuration number.
  Output:      list of words represented by the features, list of unique features.
  """
  try:
    try:
      config_num = int(input("Enter config_num: "))
      if config_num < 0 or config_num > 6: 
        global_.log_entry("Invalid configuration number.", global_.const_error)
      else: 
        main_list, feature_list = data_representation.data_representation_testing(sentence,config_num)
        print("Main List with data representation:")
        print(main_list)
        print("Unique Features:")
        print(feature_list)
    except Exception as e: 
      global_.log_entry(str(e), global_.const_error) 
      global_.log_entry("Invalid format of configuration number entered.", global_.const_error)
  except Exception as e:
    global_.log_entry("Testing terminated.", global_.const_error) 