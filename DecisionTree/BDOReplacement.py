#Bank Data Output Replacement

import DataFormatting
import Predictions
import ID3
import pandas as pd

bankTrainReplaceDF = DataFormatting.finalBankTrainReplaceDF
bankTestReplaceDF = DataFormatting.finalBankTestReplaceDF

print("Starting to build trees for bank train data with replacement")
trainReplaceTrees = pd.DataFrame(index=range(1,17),columns =['Entropy', 'ME', 'GI'])
bankTrainReplacePrediction = pd.DataFrame(index=range(1,17),columns =['Entropy', 'ME', 'GI'])
for w in range(1,4):
  print("Starting heuristic #" + str(w))
  for x in range (1,17):
    if (w == 1):
      method = 'Entropy'
    if (w == 2):
      method = 'ME'
    if (w == 3):
      method = 'GI'
    print("Building tree with max depth: " + str(x))
    tree = ID3.id3(bankTrainReplaceDF, 'y', method, x)
    #saves tree in a dataframe so that it doesn't need to be made again
    trainReplaceTrees.at[x, method] = tree
    #saves prediction value in table
    bankTrainReplacePrediction.at[x, method] = Predictions.evaluate(tree, bankTrainReplaceDF, 'y')
    #checks to see if tree has changed since the previous level
    #if not then the loop can stop making levels for the tree with that heuristic
    if x > 1:
      if trainReplaceTrees.at[x-1, method] == trainReplaceTrees.at[x, method]:
        print("The tree with max depth " + str(x) + " is the same as " + str(x-1))
        break
    

print("this is the bank train prediction table after replacement")
print(bankTrainReplacePrediction)


bankTestReplacePrediction = pd.DataFrame(index=range(1,17),columns =['Entropy', 'ME', 'GI'])
for w in range(1,4):
  for x in range (1,17):
    if (w == 1):
      method = 'Entropy'
    if (w == 2):
      method = 'ME'
    if (w == 3):
      method = 'GI'
    bankTestReplacePrediction.at[x, method] = Predictions.evaluate(trainReplaceTrees.at[x, method], bankTestReplaceDF, 'y')
    
print("this is the bank test prediction table after replacement")
print(bankTestReplacePrediction)