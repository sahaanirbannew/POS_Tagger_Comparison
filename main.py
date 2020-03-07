"""
Developer:          Anirban Saha. 
Version:            v1.0 (released)
Date:               07.03.2020
Input:              
Output:             
Description:        The function calls the training function as of now.   

Version History:
Version |   Date    |  Change ID |  Changes 
1.01                                Removed dead code. 
1.01     07.03.2020                 Added the word at the start of the sequence of DRF. 
"""
from programmes import test, train


"""
choice = input("Enter 1 for data representation for training and 2 for data representation for testing: ")
if choice == 1: 
  train.train()
if choice == 2: 
  test.test()
"""
print("*****  Guide to Dataset namea  *****")
print("penn:  for Penn Tree Bank")
print("conll: for CONLL ")
print("genia: for Genia") 
print("") 
print("*****  Guide to Configuration  *****")
print("0: word = 1, orthographic features = 0, morphological features = 1")
print("1: word = 1, orthographic features = 0, morphological features = 0")
print("2: word = 1, orthographic features = 1, morphological features = 1")
print("3: word = 1, orthographic features = 1, morphological features = 0")
print("4: word = 0, orthographic features = 0, morphological features = 1")
print("5: word = 0, orthographic features = 1, morphological features = 1")
print("6: word = 0, orthographic features = 1, morphological features = 0")
train.train()