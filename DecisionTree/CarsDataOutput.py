import Predictions
import ID3
import DataFormatting
import pandas as pd

carsTrainDF = DataFormatting.finalCarTrainDF
carsTestDF = DataFormatting.finalCarTestDF

#compares trees with depth 1-6 and uses 3 different information gain heuristics
trainTrees = pd.DataFrame(index=range(1,7),columns =['Entropy', 'ME', 'GI'])
trainPrediction = pd.DataFrame(index=range(1,7),columns =['Entropy', 'ME', 'GI'])
for y in range(1,4):
  for x in range (1,7):
    if (y == 1):
      method = 'Entropy'
    if (y == 2):
      method = 'ME'
    if (y == 3):
      method = 'GI'
    tree = ID3.id3(carsTrainDF, 'label', method, x)
    #saves tree in a dataframe so that it doesn't need to be made again
    trainTrees.at[x, method] = tree
    #saves prediction value in table
    trainPrediction.at[x, method] = Predictions.evaluate(tree, carsTrainDF, 'label')
    #checks to see if tree has changed since the previous level
    #if not then the loop can stop making levels for the tree with that heuristic
    if x > 1:
      if trainTrees.at[x-1, method] == trainTrees.at[x, method]:
        break

    
print("This is the cars train data prediction table")
print(trainPrediction)
print("\n")

testPrediction = pd.DataFrame(index=range(1,7),columns =['Entropy', 'ME', 'GI'])
for y in range(1,4):
  for x in range (1,7):
    if (y == 1):
      method = 'Entropy'
    if (y == 2):
      method = 'ME'
    if (y == 3):
      method = 'GI'
    testPrediction.at[x, method] = Predictions.evaluate(trainTrees.at[x, method], carsTestDF, 'label')
    
print("This is the cars test data prediction table")
print(testPrediction)