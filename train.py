from programmes import data_representation, global_
"""
List of valid config numbers and their interpretations: 
Config_num      Word        Orthographic Features     Morphological Features 
0               yes         no                        yes   
1               yes         no                        no
2               yes         yes                       yes                   << all features.
3               yes         yes                       no
4               no          no                        yes
5               no          no                        no
6               no          yes                       yes
7               no          yes                       no
"""
"""
List of acceptable dataset names:
- penn
- conll 
- genia
""" 

"""
Description:Prepares data representation files for training. Asks for the dataset and the configuration number.  
Input:      N.A.
Output:     N.A.
"""
def train(): 
  accepted_datasets = ["penn","conll","genia"]
  accepted_penn_tree_bank_names = ["penn tree bank", "penn tree", "penn"]
  
  dataset = input("Enter dataset name: ")
  if dataset in accepted_penn_tree_bank_names: dataset = "penn"
  if dataset not in accepted_datasets: 
    global_.log_entry("Dataset name is not recognised. Please check spelling for errors.", global_.const_error)
  else: 
    try:
      config_num = int(input("Enter config_num: "))
      if config_num < 0 or config_num > 6: 
        global_.log_entry("Invalid configuration number entered.", global_.const_error)
      else:
        file_path = "./datasets/dataset_"+dataset+".pkl"
        data_representation.genertate_drf_files(file_path, config_num) 
    except Exception as e: 
      global_.log_entry(str(e), global_.const_error) 
      global_.log_entry("Invalid format of configuration number entered.", global_.const_error) 
  