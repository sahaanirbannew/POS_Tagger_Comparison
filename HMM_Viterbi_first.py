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
        self.emissionProbDict = emissionProbDict        
        self.outgoingTagCountDict = forwardtagcount         
        fname="features/"+self.conf+"/"+"features"+"_"+self.corpus+"_train"+".pkl"
        print(fname)
        self.AllWordsDict_Inv=pickle.load(open(fname,"rb"))        
        self.AllWordsDict={value: key for key, value in self.AllWordsDict_Inv.items()} 
        #print(self.AllWordsDict)
        tname="tags/"+self.conf+"/"+"tags"+"_"+self.corpus+"_train"+".pkl"
        print(tname)
        self.tagStateDict_Inv=pickle.load(open(tname,"rb"))       
        self.tagStateDict={value: key for key, value in self.tagStateDict_Inv.items()}
        #print(self.tagStateDict)               
        values=list(self.tagStateDict.values())
        #print(values)
        values.remove("START")
        self.tagStateDict = { i : values[i] for i in range(0, len(values) ) }
        #print(self.tagStateDict)
        self.noOfTag=len(self.tagStateDict)        
        self.rawInput=pickle.load(open(rawcorpus,"rb"))
        oname="output_test/"+self.corpus+"_"+self.corpus2+"_"+self.conf+"_firstOrder_output"+".txt"
        print(oname)
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
        sentence = [x for x in rawSentence if x != []] #list of features without empty set
        #print("--------------sentence ---------------")
        #print(sentence)
        #print(len(rawSentence))
        length=len(sentence)
        #print(length)
        tagCount= int(self.noOfTag)        
        viterbi = [[0 for x in range(length)] for y in range(tagCount)]        
        backtrack = [[0 for x in range(length)] for y in range(tagCount)] 
        # THe below lines of code decodes the tags for each first word of the sentence
        
        if length!=0:
            for key in self.tagStateDict.keys():       
                list_of_emissionKey_o=ModelDecode.emission_probability_calculation(self,sentence[0],self.tagStateDict[key]) 
                #list_of_emissionKey_o=[]# for testing word only 
                emissionkey_w=sentence[0][0]+"===>"+self.tagStateDict[key]            
                #print(emissionkey)
                if sentence[0][0] not in self.AllWordsDict.values():
                    #print("word not found : " +sentence[0][0])
                    calEmission=1
                    if not list_of_emissionKey_o:
                        if wordCount[str(sentence[0])]<2:
                            #print("Rare words")
                            #print(sentence[0][0]+":"+str(wordCount[sentence[0][0]]))
                            emissionkey_rare = "_RARE_" + "===>" + self.tagStateDict[key]
                            #calEmission=1
                            if emissionkey_rare in self.emissionProbDict:
                                #print("found emission probability: " +emissionkey_rare)
                                calEmission=self.emissionProbDict[emissionkey_rare]
                                #print(calEmission)                     
                            

                    else:
                        #calEmission=1
                        for emissionkey in list_of_emissionKey_o:                    
                            if emissionkey in self.emissionProbDict.keys():
                                calEmission = (calEmission * self.emissionProbDict[emissionkey])
                elif emissionkey_w not in self.emissionProbDict.keys():                
                    #print("emissionkey key NOT found")
                    calEmission=0        
                else:
                #print("emissionkey key found")
                    calEmission=self.emissionProbDict[emissionkey_w]
                    for emissionkey in list_of_emissionKey_o:                    
                        if emissionkey in self.emissionProbDict.keys():
                            calEmission =( calEmission * self.emissionProbDict[emissionkey])
                        #else:
                            #calEmission=calEmission
                transitionkey="START"+"===>"+self.tagStateDict[key] 
                #print(transitionkey)
                #print("final emission Prob: "+str(calEmission))
                if transitionkey in self.transitionProbDict.keys(): 
                    #print("transition key found")
                    calTransition=self.transitionProbDict[transitionkey]
                #else:
                    #print("transition key NOT found")
                    #calTransition = 1 / (self.outgoingTagCountDict['START'] + int(self.noOfTag)) #prior probability calculation       
                viterbi[key][0] = calEmission * calTransition
                backtrack[key][0] = 0 
            # The below lines of code decodes the tags for each word after first word
            for l in range(1,length):#recursion step
                for tag_to in self.tagStateDict.keys():
                    for tag_from in self.tagStateDict.keys():                    
                        list_of_emissionKey_o=ModelDecode.emission_probability_calculation(self,sentence[l],self.tagStateDict[tag_to]) 
                        #list_of_emissionKey_o=[]# for testing word only 
                        emissionkey_w = sentence[l][0] + "===>" + self.tagStateDict[tag_to]
                        #print(emissionkey)
                        if sentence[l][0] not in self.AllWordsDict.values():                        
                            #print("word not found" +sentence[l][0])
                            calEmission=1
                            if not list_of_emissionKey_o:
                                if wordCount[str(sentence[l])]<2:
                                    emissionkey_rare = "_RARE_" + "===>" + self.tagStateDict[tag_to]
                                    #calEmission=1
                                    if emissionkey_rare in self.emissionProbDict:
                                        calEmission=self.emissionProbDict[emissionkey_rare]
                                        #calEmission=1
                                    

                            else:
                                #calEmission=1
                                for emissionkey in list_of_emissionKey_o:                    
                                    if emissionkey in self.emissionProbDict.keys():
                                        calEmission =( calEmission * self.emissionProbDict[emissionkey] )                 
                        elif emissionkey_w not in self.emissionProbDict.keys():
                            #print("emissionkey key NOT found")
                            calEmission =0 
                        else:
                            #print("emissionkey key found")                         
                            calEmission=self.emissionProbDict[emissionkey_w]
                            for emissionkey in list_of_emissionKey_o:                    
                                if emissionkey in self.emissionProbDict.keys():
                                    calEmission =( calEmission * self.emissionProbDict[emissionkey])
                                #else:
                                    #calEmission=calEmission
                        transitionkey = self.tagStateDict[tag_from] + "===>" + self.tagStateDict[tag_to]
                        #print(transitionkey)
                        if transitionkey in self.transitionProbDict:
                            #print("transition key NOT found")
                            #calTransition = 1 / (self.outgoingTagCountDict[self.tagStateDict[tag_from]] + int(self.noOfTag))  #prior probability calculation 
                                                
                        #else:
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
                    
            path = [str(sentence[length-1])+"===>"+self.tagStateDict[best]]        
            nice_path = [self.tagStateDict[best]]
            for t in range(length-1, 0, -1):
                best = backtrack[best][t]            
                path[0:0] = [str(sentence[t-1])+"===>"+self.tagStateDict[best]]  #   for "push"  to first in list since back tracing         
            return (path)
        else:
            path=[]
            return(path)
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



