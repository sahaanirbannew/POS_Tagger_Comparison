
"""
Developer:          Divya sasidharan 
Version:            v1.0 (released)
Date:               14.03.2020
             
Description:        Start of Accuracy Check.   
Version History:
Version |   Date    |  Change ID |  Changes 
1.01                                Initial draft version
"""
import ast
import math
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
        with open(self.actualFilePath, 'r') as f:
            print("------------Actual line------------------")
            for line in f:
                line=ast.literal_eval(line)                
                #print(line)
                for i in line:
                    value=i[1][0]+"->"+i[0]
                    self.actualLines.append(value)
            print(self.actualLines)
      

        with open(self.predictedFilePath, 'r') as f:
            print("------------ predicted line------------------")
            for line in f:
                line=ast.literal_eval(line)               
                for i in line:
                    self.predictedLines.append(i)
            print(self.predictedLines)      

        if len(self.actualLines) != len(self.predictedLines):
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
    expectedArrary = []
    actualArray = []
    for i in range(len(self.actualLines)):
        #print(self.expectedLines[i])
        currActual= self.actualLines[i].split("->")[1].strip()  
        #print(self.actualLines[i])
        currPredicted = self.predictedLines[i].split("->")[1].strip()
        totalcount += 1
        actualArrary.append(currActual)
        predictedArray.append(currPredicted)
        if currActual == currPredicted:
            totalmatch += 1
    y_pred = pd.Series(predictedArray, name='Predicted')
    y_exp = pd.Series(ActualArrary, name='Actual')
    df_confusion = pd.crosstab(y_exp, y_pred, rownames=['Actual'], colnames=['Predicted'], margins=True)
    print (df_confusion)
    print("accuracy %d/%d: %.2f%%" % (totalmatch, totalcount, 100*(totalmatch/totalcount)))

rawCorpus_test="Input_test/rawCorpus_test.txt"
rawCorpus_out="output_test/rawCorpus_out.txt"
accuracy = Accuracy(rawCorpus_test, rawCorpus_out)