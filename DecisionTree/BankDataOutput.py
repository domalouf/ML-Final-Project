import ID3
import Predictions
import DataFormatting
import pandas as pd
import statistics

bankTrainDF = DataFormatting.finalBankTrainDF
bankTestDF = DataFormatting.finalBankTestDF

print("Starting to build trees for bank train data")
trainTrees = pd.DataFrame(index=range(1,17),columns =['Entropy', 'ME', 'GI'])
bankTrainPrediction = pd.DataFrame(index=range(1,17),columns =['Entropy', 'ME', 'GI'])
for y in range(1,4):
  print("Starting heuristic #" + str(y))
  for x in range (1,17):
    if (y == 1):
      method = 'Entropy'
    if (y == 2):
      method = 'ME'
    if (y == 3):
      method = 'GI'
    print("Building tree with max depth: " + str(x))
    tree = ID3.id3(bankTrainDF, 'y', method, x)
    #saves tree in a dataframe so that it doesn't need to be made again
    trainTrees.at[x, method] = tree
    #saves predictoin value in table
    bankTrainPrediction.at[x, method] = Predictions.evaluate(tree, bankTrainDF, 'y')
    #checks to see if tree has changed since the previous level
    #if not then the loop can stop making levels for the tree with that heuristic
    if x > 1:
      if trainTrees.at[x-1, method] == trainTrees.at[x, method]:
        print("The tree with max depth " + str(x) + " is the same as " + str(x-1))
        break
    
print("this is the bank train prediction table")
print(bankTrainPrediction)

bankTestPrediction = pd.DataFrame(index=range(1,17),columns =['Entropy', 'ME', 'GI'])
for y in range(1,4):
  for x in range (1,17):
    if (y == 1):
      method = 'Entropy'
    if (y == 2):
      method = 'ME'
    if (y == 3):
      method = 'GI'
    bankTestPrediction.at[x, method] = Predictions.evaluate(trainTrees.at[x, method], bankTestDF, 'y')
    
print("this is the bank test prediction table")
print(bankTestPrediction)