import ID3
import pandas as pd

# instance is a list of values for attributes that will be passed into the tree (e.g. [‘Outlook’][‘Rain’][‘Wind’][‘Strong’])
def predict(tree, instance):
    if not isinstance(tree, dict): #if it is leaf node
        return tree #return the value
    else:
        rootNode = next(iter(tree)) #getting first key/attribute name of the dictionary
        attributeValue = instance[rootNode] #value of the attribute
        if attributeValue in tree[rootNode]: #checking the attribute value in current tree node
            return predict(tree[rootNode][attributeValue], instance) #go to next attribute
        else:
            return None
        
# returns accuracy of a decision tree tested on a dataframe
def evaluate(tree, df, label):
    correctPrediction = 0
    wrongPrediction = 0
    for index, row in df.iterrows(): #loops through rows in dataset
        result = predict(tree, df.iloc[index]) #predict the row
        if result == df[label].iloc[index]:
            correctPrediction += 1
        else:
            wrongPrediction += 1
    accuracy = correctPrediction / (correctPrediction + wrongPrediction)
    return accuracy