"""
Developer:          Divya sasidharan 
Version:            v1.0 (released)
Date:               17.08.2020
             
Description:        Start of HMM model training and decoding.   
Version History:
Version |   Date    |  Change ID |  Changes 
1.01                                Initial draft version
1.02                                Updated the calls for zero and second order
1.03                                Updated the calls for zero and second order
1.04                                Updated the code for input configuration files without words
1.05                                Updated the code for cross corpus training and testing
"""
import os
import time 
import pickle
import HMM_Training_v1 as train
import HMM_Viterbi_zero as decode_0
import HMM_Viterbi_first as decode_1 
#import hmm_viterbi_zero_v1 as decode_0
#import HMM_Viterbi_first_v2 as decode_1 # for last 3 conf
import HMM_Viterbi_second as decode_2


def open_file(configNo,corpus_name,dataset):
    """Read the corpus file based on the input corpus_name.    
    Input:-corpus name [Hint: penn or genia or conll]
    Output:- corpus file content
    """
    fileLoc="drf/"+configNo+"/"
    filelist=os.listdir(fileLoc) # returns all the pickle files in drf folder    
    for file in filelist:        
        corpus = file.split('_')[1].split('.')[0] 
        set= file.split('_')[2].split('.')[0] 
        if corpus==corpus_name and set==dataset: 
            corpus_file=pickle.load(open(fileLoc+file,"rb"))
    return corpus_file
            
         
        
        
def main(configNo,tagger,corpus_name,corpus_name_test):
    """The main function for HMM Model."""
    print("Start of HMM_Main")  
    
    tagger=tagger
    corpus_name=corpus_name
    start_time = time.time()
    if corpus_name=="penn" or corpus_name=="conll" or corpus_name=="genia":        
        if int(tagger)==0:
            print("Start of training ")  
            dataset="train"
            corpus=open_file(configNo,corpus_name,dataset)             
            tagCount_out=train.tagCount(corpus)            
            transitionProbability_out=train.transitionProbability_zeroOrder(corpus)            
            emissionProbability_out=train.emissionProbability(corpus,configNo)            
            print("Training completed!!")
            dataset="test"            
            name="drf/"+configNo+"/"+"drf"+"_"+corpus_name_test+"_"+dataset+".pkl"
            print(name)
            test=decode_0.ModelDecode(corpus_name,corpus_name_test,configNo,name,transitionProbability_out,emissionProbability_out,tagCount_out) 
            test.decode() 
        elif int(tagger)==1:
            print("Start of training for first order ")
            dataset="train"
            corpus=open_file(configNo,corpus_name,dataset) 
            #print(corpus)
            tagCount_out=train.tagCount(corpus)            
            transitionProbability_out,forwardtagcount_out=train.transitionProbability_firstOrder(corpus) 
            #print(transitionProbability_out)
            emissionProbability_out=train.emissionProbability(corpus,configNo) 
            #print(emissionProbability_out)
            print("Training completed!!") 
            dataset="test"
            name="drf/"+configNo+"/"+"drf"+"_"+corpus_name_test+"_"+dataset+".pkl"
            test=decode_1.ModelDecode(corpus_name,corpus_name_test,configNo,name,transitionProbability_out,emissionProbability_out,forwardtagcount_out) 
            test_output=test.decode()
        elif int(tagger)==2:
            print("Start of training for second order ")
            dataset="train"
            corpus=open_file(configNo,corpus_name,dataset)
            tagCount_out=train.tagCount(corpus)                                 
            transitionProbability_out,forwardtagcount_out=train.transitionProbability_secondOrder(corpus)          
            emissionProbability_out=train.emissionProbability(corpus,configNo)            
            print("Training completed!!")
            dataset="test"
            name="drf/"+configNo+"/"+"drf"+"_"+corpus_name_test+"_"+dataset+".pkl"
            test=decode_2.ModelDecode(corpus_name,corpus_name_test,configNo,name,transitionProbability_out,emissionProbability_out,forwardtagcount_out)
            test_output=test.decode()                   
        else:
            print("Not a valid input")      
        print("--- %s seconds is total execution time ---" % (time.time() - start_time))
    else:
        print("Invalid corpus name")
    
#main("conf 1",2,"penn","penn")# Example to run the code
