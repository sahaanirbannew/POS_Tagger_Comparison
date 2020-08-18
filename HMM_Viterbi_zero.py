import sys
import json
import os
import pickle
import ast

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
    def __init__(self,corpus_name,corpus_name_test,configNo, rawcorpus,transitionProbDict,emissionProbDict,forwardtagcount): 
        self.conf=configNo
        self.corpus=corpus_name
        self.corpus2=corpus_name_test
        self.transitionProbDict = transitionProbDict
        print(type(self.transitionProbDict))
        self.emissionProbDict = emissionProbDict        
        self.outgoingTagCountDict = forwardtagcount 
        fname="features/"+self.conf+"/"+"features"+"_"+self.corpus+"_train"+".pkl"
        self.AllWordsDict_Inv=pickle.load(open(fname,"rb"))        
        self.AllWordsDict={value: key for key, value in self.AllWordsDict_Inv.items()} 
        #print(self.AllWordsDict)
        tname="tags/"+self.conf+"/"+"tags"+"_"+self.corpus+"_train"+".pkl"
        self.tagStateDict_Inv=pickle.load(open(tname,"rb"))        
        self.tagStateDict={value: key for key, value in self.tagStateDict_Inv.items()}        
        values=list(self.tagStateDict.values())
        values.remove("START")
        #print(values)
        self.tagStateDict = { i : values[i] for i in range(0, len(values) ) }
        #print(self.tagStateDict)
        self.noOfTag=len(self.tagStateDict)
        self.rawInput=pickle.load(open(rawcorpus,"rb"))
        #print(self.rawInput)
        oname="output_test/"+self.corpus+"_"+self.corpus2+"_"+self.conf+"_zeroOrder_output"+".txt"
        self.outFile = open(oname,'+w')  
    
    def emission_probability_calculation(self,featureList,tag):
        list_of_emissionKey = []
        #list_of_emissionKey.append(featureList[0]+"->"+tag) 
        for f in featureList[1:]:
            list_of_emissionKey.append(f+"===>"+tag) 
        #print(list_of_emissionKey)
        return list_of_emissionKey

    

    def Viterbi(self,rawSentence,wordCount):
        """
        Description:- Decode line by line the entire corpus
        Input:- class attributes and each sentence from corpus
        output:- word-->tag for each sentence
        """
        score=0
        calEmission=0
        calTransition=0        
        #for words in rawSentence:
            #sentence.append(words)
        sentence = [x for x in rawSentence if x != []] #list of words
        #print("--------------sentence ---------------")
        #print(sentence)
        #print(len(rawSentence))
        length=len(sentence)
        #print(length)
        tagCount= int(self.noOfTag)        
        viterbi = [0 for y in range(length)]        
        backtrack = [0 for y in range(length)]
        #print(viterbi)
        #print(backtrack)
        path=[]
        

        for l in range(0,length):#recursion step
            for tag_to in self.tagStateDict.keys():
                score=0
                calEmission=0
                calTransition=0 
                
                list_of_emissionKey_o=ModelDecode.emission_probability_calculation(self,sentence[l],self.tagStateDict[tag_to]) 
                #print(list_of_emissionKey_o)
                #list_of_emissionKey_o=[]# for testing word only 
                emissionkey_w = sentence[l][0] + "===>" + self.tagStateDict[tag_to]
                #print(emissionkey_w)
                
                if sentence[l][0] not in self.AllWordsDict.values(): 
                    #print("word not found in features")
                    if not list_of_emissionKey_o:
                        if wordCount[str(sentence[l])]<2:
                            #print("word is RARE")
                            emissionkey_rare = "_RARE_" + "===>" + self.tagStateDict[tag_to]
                            if emissionkey_rare in self.emissionProbDict:
                                calEmission=self.emissionProbDict[emissionkey_rare]
                            for emissionkey in list_of_emissionKey_o:
                                if emissionkey in self.emissionProbDict.keys():
                                    calEmission = calEmission * self.emissionProbDict[emissionkey]                           
                        
                    else:
                        calEmission=1                        
                        for emissionkey in list_of_emissionKey_o:                    
                            if emissionkey in self.emissionProbDict.keys():
                                calEmission = calEmission * self.emissionProbDict[emissionkey]
                            
                                                   
                elif emissionkey_w not in self.emissionProbDict.keys():#emission key not found  
                    #print("emissionkey not found")
                    calEmission =0 
                else:                       
                    calEmission=self.emissionProbDict[emissionkey_w]
                    #print("emission key found "+emissionkey_w)
                    #print(list_of_emissionKey_o)
                    for emissionkey in list_of_emissionKey_o:                    
                        if emissionkey in self.emissionProbDict.keys():
                            calEmission = calEmission * self.emissionProbDict[emissionkey]
                transitionkey =self.tagStateDict[tag_to]                
                calTransition = self.transitionProbDict[transitionkey]                
                score = calTransition * calEmission                
                if score > viterbi[l]:                    
                    viterbi[l] = score 
                    backtrack[l] = tag_to
                else:
                    continue
            
        for t in range(length,0, -1):
            best = backtrack[t-1]            
            path[0:0] = [str(sentence[t-1])+"===>"+self.tagStateDict[best]]  #   for "push"  to first in list since back tracing    
        #print(path)
        return (path)

    def decode(self):
        """
        Description:- Start of decoding .Calls viterbi() method for each sentence in corpus
        Input:- class attributes 
        output:- word-->tag output from viterbi() for each sentence is written into file 
        """

        print("start decoding")       
        listToStr=[]
        start=0 
        wordCount={}
        for line in self.rawInput:             
            #line=ast.literal_eval(line)            
            #for l in line:
            #print(line)
            if str(line[2]) in wordCount:
                wordCount[str(line[2])]=wordCount[str(line[2])]+1
            else:
                wordCount[str(line[2])]=1
        #print(wordCount)     
        for line in  self.rawInput: 
            if "START" in line[0]:                
                if len(listToStr)!=0:
                    #print("---line---")
                    #print(listToStr)
                    path=ModelDecode.Viterbi(self,listToStr,wordCount) 
                    #print(path)                    
                    outputSentence = '$$$$'
                    outputSentence = outputSentence + str(path)
                    outputSentence = outputSentence.strip('$$$$')    
                    outputSentence = outputSentence + '\n'
                    self.outFile.write(outputSentence)                 
                listToStr=[]                    
                listToStr.append(line[2])
                
            else:
                listToStr.append(line[2])
        #print("----line----")
        #print(listToStr)
        path=ModelDecode.Viterbi(self,listToStr,wordCount)  
        #print(path)
        outputSentence = '$$$$'
        outputSentence = outputSentence + str(path)
        outputSentence = outputSentence.strip('$$$$') 
        outputSentence = outputSentence + '\n'
        self.outFile.write(outputSentence)                 
        print("end of decoding")



