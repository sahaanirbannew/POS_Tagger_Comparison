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
        self.AllWordsDict_Inv=pickle.load(open(fname,"rb"))        
        self.AllWordsDict={value: key for key, value in self.AllWordsDict_Inv.items()} 
        #print(self.AllWordsDict)
        tname="tags/"+self.conf+"/"+"tags"+"_"+self.corpus+"_train"+".pkl"
        self.tagStateDict_Inv=pickle.load(open(tname,"rb"))        
        self.tagStateDict={value: key for key, value in self.tagStateDict_Inv.items()}  
        #self.tagStateDict.pop(7)
        values=list(self.tagStateDict.values())
        values.remove("START")
        values.append("START")
        values.append("STOP")                
        self.tagStateDict = { i : values[i] for i in range(0, len(values) ) }        
        self.noOfTag=len(self.tagStateDict)        
        self.rawInput=pickle.load(open(rawcorpus,"rb"))
        oname="output_test/"+self.corpus+"_"+self.corpus2+"_"+self.conf+"_secondOrder_output"+".txt"
        print(oname)
        self.outFile = open(oname,'+w')  
    
    def emission_probability_calculation(self,featureList,tag):
        list_of_emissionKey = []
        #list_of_emissionKey.append(featureList[0]+"===>"+tag) 
        for f in featureList[1:]:
            list_of_emissionKey.append(f+"===>"+tag) 
        #print(list_of_emissionKey)
        return list_of_emissionKey

 


    def Viterbi(self,rawSentence,wordCount):
            """
            Description:- Decode line by line the entire corpus
            Input:- class attributes and each sentence from corpus
            output:- word-===>tag for each sentence
            """
            
            
                  
            words = [x for x in rawSentence if x != []] #list of words
            #print("--------------sentence ---------------")
            #print(sentence)
            #print(len(rawSentence))
            length=len(words)
            tagCount= int(self.noOfTag)
            pimatrix = [[[0 for x in range(tagCount)] for x in range(tagCount)]for x in range(length + 1)]
            #print(pimatrix)
            bp = [[[0 for x in range(tagCount)] for x in range(tagCount)]for x in range(length + 1)]
            #print(bp)
            for i in range(tagCount):
                for j in range(tagCount):
                    pimatrix[0][i][j] = 0
            pimatrix[0][tagCount-2][tagCount-2] = 1
            
            y = [0 for x in range (length+1)]
            #print(pimatrix)
            for k in range(1,length+1):
                for u in range(tagCount):
                    for v in range(tagCount):
                        emissionkey_w = words[k-1][0] + "===>" + self.tagStateDict[v]                        
                        list_of_emissionKey_o=[]
                        list_of_emissionKey_o=ModelDecode.emission_probability_calculation(self,words[k-1],self.tagStateDict[v])
                        pimatrix[k][u][v] = 0
                        arg = 0
                        for w in range(tagCount):
                            temp = 0                            
                            transitionkey = self.tagStateDict[w] + "===>" + self.tagStateDict[u]+"===>"+ self.tagStateDict[v] 
                            if transitionkey in self.transitionProbDict:
                                calTransition = self.transitionProbDict[transitionkey]                            
                                if words[k-1][0] not in self.AllWordsDict.values():
                                    calEmission=1
                                    if not list_of_emissionKey_o:
                                        if wordCount[str(words[k-1])]<2:
                                            emissionkey_rare = "_RARE_" + "===>" + self.tagStateDict[v]
                                            if emissionkey_rare in self.emissionProbDict:
                                                calEmission=self.emissionProbDict[emissionkey_rare]
                                    else:
                                        for emissionkey in list_of_emissionKey_o:
                                             if emissionkey in self.emissionProbDict.keys():
                                                #calEmission = calEmission+(calEmission * self.emissionProbDict[emissionkey])
                                                calEmission = (calEmission * self.emissionProbDict[emissionkey])
                                             
                                elif emissionkey_w not in self.emissionProbDict.keys():
                                    calEmission =0
                                else:
                                    #print("emissionkey key found")                         
                                    calEmission=self.emissionProbDict[emissionkey_w]
                                    for emissionkey in list_of_emissionKey_o:                    
                                        if emissionkey in self.emissionProbDict.keys():
                                            #calEmission = calEmission+(calEmission * self.emissionProbDict[emissionkey])
                                            calEmission = (calEmission * self.emissionProbDict[emissionkey])
                            
                            
                                temp=pimatrix[k-1][w][u] * calTransition * calEmission
                                if(temp > pimatrix[k][u][v]):                                    
                                    bp[k][u][v] = w
                                    pimatrix[k][u][v] = temp
                                                       

                            
            #print(bp)
            #print(pimatrix)
            
            curr = 0
            for u in range(tagCount):
                for v in range(tagCount):
                    transitionkey = self.tagStateDict[u] + "===>" + self.tagStateDict[v]+"===>"+ 'STOP'
                    if (transitionkey in self.transitionProbDict):
                        temp = pimatrix[length][u][v]*self.transitionProbDict[transitionkey]
                        if(temp > curr):
                            curr = temp
                            y[length-1] = u
                            y[length] = v
                   
            for k in list(reversed(range(1,length-1))):
                y[k] = bp[k+2][y[k+1]][y[k+2]]    
            return y 

    def decode(self):
        """
        Description:- Start of decoding .Calls viterbi() method for each sentence in corpus
        Input:- class attributes 
        output:- word-===>tag output from viterbi() for each sentence is written into file 
        """
        print("start decoding")       
        listToStr=[]
        start=0 
        wordCount={}
        #print(self.AllWordsDict.values())
        for line in self.rawInput:            
            if str(line[2]) in wordCount:
                wordCount[str(line[2])]=wordCount[str(line[2])]+1
            else:
                wordCount[str(line[2])]=1
        for line in  self.rawInput: 
            if "START" in line[0]:                
                if len(listToStr)!=0:
                    #print("---line---")
                    #print(listToStr)
                    path=ModelDecode.Viterbi(self,listToStr,wordCount) 
                    #print(path) 
                    outputSentence=[] 
                    for j in range(1,len(path)): 
                        outputSentence.append(str(listToStr[j-1]) + "===>" + self.tagStateDict[path[j]])
                    self.outFile.write(str(outputSentence))
                    #print(outputSentence)
                    self.outFile.write('\n')
                listToStr=[]                    
                listToStr.append(line[2])
                
            else:
                listToStr.append(line[2])
        #print("----line----")
        #print(listToStr)
        path=ModelDecode.Viterbi(self,listToStr,wordCount)  
        #print(path)
        outputSentence=[]
        for j in range(1,len(path)): 
            outputSentence.append(str(listToStr[j-1]) + "===>" + self.tagStateDict[path[j]])
        self.outFile.write(str(outputSentence))
        self.outFile.write('\n')                 
        print("end of decoding")


