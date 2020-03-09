import sys
import json
import os
import pickle

class ModelDecode:
    """  

    Attributes
    ----------
    rawcorpus : file 
        specify the location of the raw corpus for decoding eg:-"Input_test/rawCorpus.txt"
    transitionProbDict : Dictionary
        contains the transition probability values from training
    emissionProbDict : Dictionary
        contains the emission probability values from training
    forwardtagcount : Dictionary
        contains the tag count values from training

    Methods
    -------
    decode(self)
        Starts the decoding  by passing each line from input raw corpus to Viterbi() method.
    Viterbi(self, rawSentence)
        Starts the decoding line by line for the input raw corpus.
    """
    def __init__(self, rawcorpus,transitionProbDict,emissionProbDict,forwardtagcount):        
        self.transitionProbDict = transitionProbDict
        self.emissionProbDict = emissionProbDict        
        self.outgoingTagCountDict = forwardtagcount         
        self.AllWordsDict_Inv=pickle.load(open("features/features_penn.pkl","rb"))        
        self.AllWordsDict={value: key for key, value in self.AllWordsDict_Inv.items()} 
        print(self.AllWordsDict)
        self.tagStateDict_Inv=pickle.load(open("tags/tags_penn.pkl","rb"))        
        self.tagStateDict={value: key for key, value in self.tagStateDict_Inv.items()}
        print(self.tagStateDict)
        self.noOfTag=len(self.tagStateDict)
        self.rawInput=open(rawcorpus,'+r')
        self.outFile = open('output_test/rawCorpus_out.txt','+w')  
    
    def emission_probability_calculation(self,word,n,tag):# replace with "get_list_of_words_encrypted_str()" method from data_representation.py
        list_of_emissionKey = []
        list_of_emissionKey.append(word+"->"+tag)
        # get Suffix
        counter = -1
        no_of_characters = len(word) * -1
        while(counter > no_of_characters):
            suffix = '*'+word[-counter:]+"->"+tag
            list_of_emissionKey.append(suffix)
            counter = counter - 1   
        # get Prefix
        counter = 1
        no_of_characters = len(word)
        while(counter < no_of_characters):
            prefix = word[0:counter]+'*'+"->"+tag
            list_of_emissionKey.append(prefix)
            counter = counter + 1
        return list_of_emissionKey

    def Viterbi(self,rawSentence):
        """
        Description:- Decode line by line the entire corpus
        Input:- class attributes and each sentence from corpus
        output:- word-->tag for each sentence
        """
        score=0
        calEmission=0
        calTransition=0
        sentence = rawSentence.split() #list of words
        length=len(sentence)
        tagCount= int(self.noOfTag) + 1        
        viterbi = [[0 for x in range(length)] for y in range(tagCount)]        
        backtrack = [[0 for x in range(length)] for y in range(tagCount)] 
        # THe below lines of code decodes the tags for each first word of the sentence
        for key in self.tagStateDict.keys():             
            list=ModelDecode.emission_probability_calculation(self,sentence[0],3,self.tagStateDict[key]) #replace with "get_list_of_words_encrypted_str()" method from data_representation.py
            
            emissionkey_w=sentence[0]+"->"+self.tagStateDict[key]            
            #print(emissionkey)
            if sentence[0] not in self.AllWordsDict.values():
                calEmission=1
                for emissionkey in list:                    
                    if emissionkey in self.emissionProbDict.keys():
                        calEmission = calEmission * self.emissionProbDict[emissionkey]
                    else:
                        calEmission=calEmission
                #print("word NOT found")
            elif emissionkey_w not in self.emissionProbDict.keys():                
                #print("emissionkey key NOT found")
                calEmission=0        
            else:
               #print("emissionkey key found")
                calEmission=1
                for emissionkey in list:                    
                    if emissionkey in self.emissionProbDict.keys():
                        calEmission = calEmission * self.emissionProbDict[emissionkey]
                    else:
                        calEmission=calEmission
            transitionkey="START"+"->"+self.tagStateDict[key] 
            #print(transitionkey)
            if transitionkey in self.transitionProbDict.keys(): 
                #print("transition key found")
                calTransition=self.transitionProbDict[transitionkey]
            else:
                #print("transition key NOT found")
                calTransition = 1 / (self.outgoingTagCountDict['START'] + int(self.noOfTag)) #prior probability calculation       
            viterbi[key][0] = calEmission * calTransition
            backtrack[key][0] = 0  
        # The below lines of code decodes the tags for each word after first word
        for l in range(1,length):#recursion step
            for tag_to in self.tagStateDict.keys():
                for tag_from in self.tagStateDict.keys():                    
                    list=ModelDecode.emission_probability_calculation(self,sentence[l],3,self.tagStateDict[tag_to]) #replace with "get_list_of_words_encrypted_str()" method from data_representation.py
                    emissionkey_w = sentence[l] + "->" + self.tagStateDict[tag_to]
                    #print(emissionkey)
                    if sentence[l] not in self.AllWordsDict.values():                        
                        calEmission=1
                        for emissionkey in list:                    
                            if emissionkey in self.emissionProbDict.keys():
                                calEmission = calEmission * self.emissionProbDict[emissionkey]
                            else:
                                calEmission=calEmission
                        #print("word NOT found")                        
                    elif emissionkey_w not in self.emissionProbDict.keys():
                        #print("emissionkey key NOT found")
                        calEmission =0 
                    else:
                         #print("emissionkey key found")                         
                         calEmission=1
                         for emissionkey in list:                    
                            if emissionkey in self.emissionProbDict.keys():
                                calEmission = calEmission * self.emissionProbDict[emissionkey]
                            else:
                                calEmission=calEmission
                    transitionkey = self.tagStateDict[tag_from] + "->" + self.tagStateDict[tag_to]
                    #print(transitionkey)
                    if transitionkey not in self.transitionProbDict:
                        #print("transition key NOT found")
                        calTransition = 1 / (self.outgoingTagCountDict[self.tagStateDict[tag_from]] + int(self.noOfTag))  #prior probability calculation 
                                            
                    else:
                        #print("transition key found")
                        calTransition = self.transitionProbDict[transitionkey]
                    score = viterbi[tag_from][l-1] * calTransition * calEmission
                    if score > viterbi[tag_to][l]:
                        viterbi[tag_to][l] = score
                        backtrack[tag_to][l] = tag_from
                    else:
                        continue
        best = 0
        for i in self.tagStateDict.keys():
            if viterbi[i][length-1] > viterbi[best][length-1]: # get max viterbi[state,Word]
                best = i
                
        path = [sentence[length-1]+"->"+self.tagStateDict[best]]        
        nice_path = [self.tagStateDict[best]]
        for t in range(length-1, 0, -1):
            best = backtrack[best][t]            
            path[0:0] = [sentence[t-1]+"->"+self.tagStateDict[best]]  #   for "push"  to first in list since back tracing         
        return (path)

    def decode(self):
        """
        Description:- Start of decoding .Calls viterbi() method for each sentence in corpus
        Input:- class attributes 
        output:- word-->tag output from viterbi() for each sentence is written into file 
        """
        print("start decoding")
        for line in self.rawInput:
            path=ModelDecode.Viterbi(self,line)  
            #print(path)
            #self.outFile.write(str(path))            
            outputSentence = '$$$$'
            for i in path:        
                outputSentence = outputSentence + i+ ' '
            outputSentence = outputSentence.strip('$$$$')    
            outputSentence = outputSentence + '\n'
            self.outFile.write(outputSentence) 
            #break
        print("end of decoding")



