import pickle
import time
#import hmm_decode_withFeatures_2 as decode

# global variable declaration

tags={}
forwardtagcount={}
transition_firstOrder={}
transition_secondOrder={}
transition_secondOrder_p={}
transitionProbabilities_zeroOrder={}
transitionProbabilities_firstOrder={}
transitionProbabilities_secondOrder={}
featureCount={}
emissionProbabilities={}

def tagCount(corpus):
    """
    Description:- Build the dictionary with key as "tag" and value =count of key in corpus 
    Input :- corpus data read from corpus files 
    Output :- Dictionary with tag count
    """
    tags["START"]=0 
    for i in corpus:
        tag=i[0] 
        if "SOS" in i[1]:            
            tags["START"]=tags["START"]+1        
        if tag in tags:
           tags[tag]=tags[tag]+1
        else:
           tags[tag]=1    
    return tags

def tagCount_secondOrder(corpus):    
    """
    Description:- Build the dictionary with key as 'previous_previous_tag->previous_tag' and value= count of key in corpus.
    This is for transitionProbability_secondOrder calculation
    Input :- corpus data read from corpus files 
    Output :- Dictionary with tag count for second order
    """
    for previous_previous, previous in zip(corpus, corpus[1:]):        
        transtioninfo=previous_previous[0]+"->"+previous[0] # key for transition dictionary
        
        if transtioninfo in transition_secondOrder_p:
            transition_secondOrder_p[transtioninfo]=transition_secondOrder_p[transtioninfo]+1 
        else:
            transition_secondOrder_p[transtioninfo]=1 


def transitionProbability_secondOrder(corpus): 
    """
    Description:- Transition Probability calculation  for second order.
    Transition Probability= {C(t(i-2)->t(i-1)->t(i))/C(t(i-2)->t(i-1))} 
    Input:-corpus data read from corpus files
    Output:- transitionProbabilities for secondOrder
    """
    for previous_previous,previous, current in zip(corpus, corpus[1:],corpus[2:]): # calculating tag count
        transtioninfo=previous_previous[0]+"->"+previous[0]+"->"+current[0] # key for transition dictionary        
        if transtioninfo in transition_secondOrder:
            transition_secondOrder[transtioninfo]=transition_secondOrder[transtioninfo]+1 
        else:
            transition_secondOrder[transtioninfo]=1      
    for transitionInfo in transition_secondOrder: 
        previous_previousTag=transitionInfo.split("->")[0].strip()
        previousTag=transitionInfo.split("->")[1].strip()
        Tag=previous_previousTag+"->"+previousTag
        transitionProbabilities_secondOrder[transitionInfo]=(transition_secondOrder[transitionInfo])/(transition_secondOrder_p[Tag])  


def transitionProbability_firstOrder(corpus):  
    """
    Description:-Transition Probability calculation  for first order.Transition Probability= {C(t(i-1)->t(i))/C(t(i-1))}
    Input:- corpus data read from corpus files
    Output:- transitionProbabilities for firstOrder     
    """
    for i in corpus:
        tag=i[0] 
        if "SOS" in i[1]:
            transtioninfo="START"+"->"+tag
            if transtioninfo in transition_firstOrder:
                transition_firstOrder[transtioninfo]=transition_firstOrder[transtioninfo]+1 
            else:
                transition_firstOrder[transtioninfo]=1 
            previous="START"
            if previous in forwardtagcount:
                forwardtagcount[previous]= forwardtagcount[previous]+1            
            else:
                forwardtagcount[previous]=1
    for previous, current in zip(corpus, corpus[1:]):  # calculating tag count        
        if "SOS" not in current[1]:
            transtioninfo=previous[0]+"->"+current[0] # key for transition dictionary
            if transtioninfo in transition_firstOrder:
                transition_firstOrder[transtioninfo]=transition_firstOrder[transtioninfo]+1 
            else:
                transition_firstOrder[transtioninfo]=1  
            if previous[0] in forwardtagcount:
                forwardtagcount[previous[0]]= forwardtagcount[previous[0]]+1            
            else:
                forwardtagcount[previous[0]]=1    
    #Transition Probability calculation
    for transitionInfo in transition_firstOrder: 
        previousTag=transitionInfo.split("->")[0].strip()
        if forwardtagcount[previousTag]>0:
            transitionProbabilities_firstOrder[transitionInfo]=(transition_firstOrder[transitionInfo])/(forwardtagcount[previousTag])
    print(forwardtagcount)
    return transitionProbabilities_firstOrder,forwardtagcount

def transitionProbability_zeroOrder(corpus):
    """
    Description:-Transition Probability calculation  for zero order.Transition Probability= {C(t(i))/N)}
    Input:- corpus data read from corpus files
    Output:- transitionProbabilities for zeroOrder     
    """
    totalCount_N=sum(tags.values())# calculating total count N
    #Transition Probability calculation  
    for tag in tags:
        count=tags[tag]
        transitionProbabilities_zeroOrder[tag]=(count)/(totalCount_N)

def emissionProbability(corpus):
    """
    Description:- Emission Probablity calculation for word->tag. Emission Probability= {C(word->t(i))/C(t(i))}
    Input:- corpus data read from corpus files
    Output:- emissionProbabilities     
    """
    for i in corpus:  # calculating word->Tag count
        for j in range(0,len(i[1])):
            featureTag=i[1][j]+"->"+i[0]
            if featureTag not in featureCount:
                featureCount[featureTag]=1
            else:
                featureCount[featureTag]=featureCount[featureTag]+1    
    #Calculating Emission Probability
    for i in featureCount:        
        tag=i.split("->")[1].strip()  
        emissionProbabilities[i]=featureCount[i]/tags[tag]
    return emissionProbabilities






