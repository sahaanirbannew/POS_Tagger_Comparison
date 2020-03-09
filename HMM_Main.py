"""
Developer:          Divya sasidharan 
Version:            v1.0 (released)
Date:               09.03.2020
             
Description:        Start of HMM model training.   
Version History:
Version |   Date    |  Change ID |  Changes 
1.01                                Initial draft version
"""

import os
import HMM_Training as train
import time 
import pickle
import  HMM_Viterbi as decode
import data_representation  as dr

def open_file(corpus_name):
    """Read the corpus file based on the input corpus_name.    
    Input:-corpus name [Hint: penn or genia or conll]
    Output:- corpus file content
    """
    filelist=os.listdir("drf/") # returns all the pickle files in drf folder    
    for file in filelist:        
        corpus = file.split('_')[1].split('.')[0]        
        if corpus==corpus_name: 
            corpus_file=pickle.load(open("drf/"+file,"rb"))
    return corpus_file
            
            
        
        
def main(tagger,corpus_name):
    """The main function for HMM Model."""
    print("Start of HMM_Main")    
    #tagger=input(" enter the tagger type [Hint: 0= zero order/ 1=First order/2=Second order] :")
    #corpus_name=input(" enter the corpus name[Hint: penn or genia or conll]:") 
    tagger=tagger
    corpus_name=corpus_name
    start_time = time.time()
    if corpus_name=="penn" or corpusName=="conll" or corpusName=="genia":        
        if int(tagger)==0:
            print("Start of training for zero order")  
            print("Training completed!!")
        elif int(tagger)==1:
            print("Start of training for first order ")
            corpus=open_file(corpus_name)
            tagCount_out=train.tagCount(corpus) 
            transitionProbability_out,forwardtagcount_out=train.transitionProbability_firstOrder(corpus) 
            print(forwardtagcount_out)
            emissionProbability_out=train.emissionProbability(corpus) 
            print("Training completed!!")         
            test=decode.ModelDecode("Input_test/rawCorpus.txt",transitionProbability_out,emissionProbability_out,forwardtagcount_out) 
            test_output=test.decode()
        elif int(tagger)==2:
            print("Start of training for second order ")
            print("Training completed!!")
        else:
            print("Not a valid input")      
        print("--- %s seconds is total execution time ---" % (time.time() - start_time))
    else:
        print("Invalid corpus name")
    #return test_output

output=main(1,"penn")
print(output)
