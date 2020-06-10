"""
Developer:          Divya sasidharan 
Version:            v1.0 (released)
Date:               10.06.2020
             
Description:        Start of Accuracy Check.   
Version History:
Version |   Date    |  Change ID |  Changes 
1.01                                Initial draft version
1.02                                updated pickle file reader
1.03                                Updated the code for input configuration files without words
1.04                                Update for F1Score,Recall,Precision using Sklearn
"""
import ast
import math
import pickle
import string
import pandas as pd


class Accuracy:
  """
    Description:- Measure accuracy of prediction
    Input :- rawCorpus_test and rawCorpus_out for comparison 
    Output :- Accuracy and confusion matrix
   """
  def __init__(self, actual, predicted):
      self.actualFilePath = actual
      self.predictedFilePath = predicted
      self.actualLines = []
      self.predictedLines = []
      if self.readfiles():
        self.calculateAccuracy()

  """
  Load the data from expected and actual files
  """
  def readfiles(self): 
        """
    Description:- read files for actual and predicted data
    Input :- rawCorpus_test and rawCorpus_out for comparison 
    Output :- True or False 
   """
        f=pickle.load(open(self.actualFilePath,"rb"))
        print("------------Actual line------------------")
        print(self.actualFilePath)
        for line in f:            
            if line[2]!=[]:
                value=str(line[2])+"===>"+line[1]
                self.actualLines.append(value)
        #print(self.actualLines)

      
        with open(self.predictedFilePath, 'r') as f:
            print("------------ predicted line------------------")
            #print(self.predictedFilePath)
            for line in f:
                line=ast.literal_eval(line)               
                for i in line:
                    self.predictedLines.append(i)
            #print(self.predictedLines)      

        if len(self.actualLines) != len(self.predictedLines):
            #print(self.actualLines)
            #print(self.predictedLines)
            print('ERROR: Expected and actual file lengths dont match')            
            return False

        return True

 
  def calculateAccuracy(self):
    """
    Description:- calculate accuracy
    Input :- actual tags and predicted tags 
    Output :- Accuracy and confusion matrix 
    """
    totalcount = 0
    totalmatch = 0
    predictedArray = []
    actualArray = []
    for i in range(len(self.actualLines)):
        #print(self.actualLines[i])
        
        currActual= self.actualLines[i].split("===>")[1].strip()         
        currPredicted = self.predictedLines[i].split("===>")[1].strip()
        totalcount += 1
        actualArray.append(currActual)
        predictedArray.append(currPredicted)
        if currActual == currPredicted:
            totalmatch += 1
    y_pred = pd.Series(predictedArray, name='Predicted')
    y_exp = pd.Series(actualArray, name='Actual')
    df_confusion = pd.crosstab(y_exp, y_pred, rownames=['Actual'], colnames=['Predicted'], margins=True)
    print (df_confusion)
    print("accuracy %d/%d: %.2f%%" % (totalmatch, totalcount, 100*(totalmatch/totalcount)))

    from sklearn.metrics import classification_report,accuracy_score,precision_score,recall_score,f1_score
    print('Flat Accuracy  :',accuracy_score(actualArray, predictedArray))
    print("Avg Precision  : " ,precision_score(actualArray, predictedArray, average='weighted'))    
    print("Avg Recall :  " ,recall_score(actualArray, predictedArray, average='weighted'))
    print("Avg F1-score :  " ,f1_score(actualArray, predictedArray, average='weighted'))
    
    

    print ('Report : ')
    print (classification_report(actualArray, predictedArray))

#accuracy = Accuracy(rawCorpus_test, rawCorpus_out)
