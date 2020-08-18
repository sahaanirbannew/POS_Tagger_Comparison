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
Countword={}

def tagCount(corpus):
    """
    Description:- Build the dictionary with key as "tag" and value =count of key in corpus 
    Input :- corpus data read from corpus files 
    Output :- Dictionary with tag count
    """
    tags["START"]=0 
    for i in corpus:
        #print(i)
        tag=i[1] 
        if "START" in i[0]:            
            tags["START"]=tags["START"]+1        
        if tag in tags:
           tags[tag]=tags[tag]+1
        else:
           tags[tag]=1    
    return tags



def transitionProbability_secondOrder(corpus): 
    """
    Description:- Transition Probability calculation  for second order.
    Transition Probability= {C(t(i-2)->t(i-1)->t(i))/C(t(i-2)->t(i-1))} 
    Input:-corpus data read from corpus files
    Output:- transitionProbabilities for secondOrder
    """
    for previous, current in zip(corpus, corpus[1:]):  
        #print(previous)
        #print(current)
        if ("START" in previous[0]):           
            transtioninfo="START"+"===>"+previous[0]+"===>"+previous[1]
            transtioninfo_s=previous[0]+"===>"+previous[1]+"===>"+current[1]
        else:
            if ("START" not in current[0]):
                transtioninfo=previous[0]+"===>"+previous[1]+"===>"+current[1]
            else:
                transtioninfo=previous[0]+"===>"+previous[1]+"===>"+"STOP"
        #print(transtioninfo)

        if transtioninfo in transition_secondOrder:
            transition_secondOrder[transtioninfo]=transition_secondOrder[transtioninfo]+1 
        else:
            transition_secondOrder[transtioninfo]=1

        if transtioninfo_s in transition_secondOrder:
            transition_secondOrder[transtioninfo_s]=transition_secondOrder[transtioninfo_s]+1 
        else:
            transition_secondOrder[transtioninfo_s]=1
    #print(transition_secondOrder)


    for previous, current in zip(corpus, corpus[1:]):  
        #print(previous)
        #print(current)
        if ("START" in previous[0]):           
            transtioninfo="START"+"===>"+previous[0] 
            transtioninfo_s=previous[0]+"===>"+previous[1]
        else:
            if ("START" not in current[0]):
                transtioninfo=previous[0]+"===>"+previous[1]
            else:
                transtioninfo=previous[0]+"===>"+previous[1]            
        #print(transtioninfo)

        if transtioninfo in transition_secondOrder_p:
            transition_secondOrder_p[transtioninfo]=transition_secondOrder_p[transtioninfo]+1 
        else:
            transition_secondOrder_p[transtioninfo]=1

        if transtioninfo_s in transition_secondOrder_p:
            transition_secondOrder_p[transtioninfo_s]=transition_secondOrder_p[transtioninfo_s]+1 
        else:
            transition_secondOrder_p[transtioninfo_s]=1
    #print(transition_secondOrder_p)

    for transitionInfo in transition_secondOrder:
        previous_previousTag=transitionInfo.split("===>")[0].strip()
        previousTag=transitionInfo.split("===>")[1].strip()
        Tag=previous_previousTag+"===>"+previousTag
        transitionProbabilities_secondOrder[transitionInfo]=(transition_secondOrder[transitionInfo])/(transition_secondOrder_p[Tag])
    #print(transition_secondOrder)
    return transitionProbabilities_secondOrder,transition_secondOrder




def transitionProbability_firstOrder(corpus):  
    """
    Description:-Transition Probability calculation  for first order.Transition Probability= {C(t(i-1)->t(i))/C(t(i-1))}
    Input:- corpus data read from corpus files
    Output:- transitionProbabilities for firstOrder     
    """
    for i in corpus:
        tag=i[1] 
        if "START" in i[0]:
            transtioninfo="START"+"===>"+tag
            if transtioninfo in transition_firstOrder:
                transition_firstOrder[transtioninfo]=transition_firstOrder[transtioninfo]+1 
            else:
                transition_firstOrder[transtioninfo]=1 
            previous=i[0]
            if previous in forwardtagcount:
                forwardtagcount[previous]= forwardtagcount[previous]+1            
            else:
                forwardtagcount[previous]=1
    for previous, current in zip(corpus, corpus[1:]):  # calculating tag count        
        if "START" not in current[0]:
            transtioninfo=previous[1]+"===>"+current[1] # key for transition dictionary
            if transtioninfo in transition_firstOrder:
                transition_firstOrder[transtioninfo]=transition_firstOrder[transtioninfo]+1 
            else:
                transition_firstOrder[transtioninfo]=1  
            if previous[1] in forwardtagcount:
                forwardtagcount[previous[1]]= forwardtagcount[previous[1]]+1            
            else:
                forwardtagcount[previous[1]]=1    
    #Transition Probability calculation
    for transitionInfo in transition_firstOrder: 
        previousTag=transitionInfo.split("===>")[0].strip()
        if forwardtagcount[previousTag]>0:
            transitionProbabilities_firstOrder[transitionInfo]=(transition_firstOrder[transitionInfo])/(forwardtagcount[previousTag])
    #print(forwardtagcount)
    return transitionProbabilities_firstOrder,forwardtagcount

def transitionProbability_zeroOrder(corpus):
    """
    Description:-Transition Probability calculation  for zero order.Transition Probability= {C(t(i))/N)}
    Input:- corpus data read from corpus files
    Output:- transitionProbabilities for zeroOrder     
    """
    totalCount_N=sum(tags.values())# calculating total count N
    print("total count is "+str(totalCount_N))
    print(type(transitionProbabilities_zeroOrder))
    #Transition Probability calculation  
    for tag in tags:
        count=tags[tag]
        value=count/totalCount_N
        if tag not in transitionProbabilities_zeroOrder:
            transitionProbabilities_zeroOrder[tag]=value   
    return transitionProbabilities_zeroOrder



def emissionProbability(corpus,configNo):
    """
    Description:- Emission Probablity calculation for word->tag. Emission Probability= {C(word->t(i))/C(t(i))}
    Input:- corpus data read from corpus files
    Output:- emissionProbabilities     
    """
    print(configNo)
    if configNo in ["conf 0" ,"conf 1", "conf 2", "conf 3","conf 4","conf 5","conf 6"]:
        print(configNo)
        wordCount={}
        for i in corpus:
            #print(i[2])
            if str(i[2]) in wordCount:
                wordCount[str(i[2])]=wordCount[str(i[2])]+1
            else:
                wordCount[str(i[2])]=1
    
        print("count for _RARE_")
        for i in corpus:  # calculating word->Tag count for RARE words
            if wordCount[str(i[2])]<2:
                featureTag="_RARE_"+"===>"+i[1]
                if featureTag not in featureCount:
                    featureCount[featureTag]=1
                else:
                    featureCount[featureTag]=featureCount[featureTag]+1  
    
    print("count for features")
    for i in corpus:  # calculating word->Tag count
        for j in range(0,len(i[2])):
            featureTag=i[2][j]+"===>"+i[1]
            if featureTag not in featureCount:
                featureCount[featureTag]=1
            else:
                featureCount[featureTag]=featureCount[featureTag]+1 
                
    print("emissionProbability calculation")
    #Calculating Emission Probability
    for i in featureCount: 
        #print(i)
        tag=i.split("===>")[1].strip()  
        emissionProbabilities[i]=featureCount[i]/tags[tag]
    return emissionProbabilities




