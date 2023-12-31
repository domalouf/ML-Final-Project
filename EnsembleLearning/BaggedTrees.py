import ID3
import pandas
import random
import math
import numpy
import DataFormatting

def baggedTrees(_df, label, T):
    #given training set with m examples
    df = _df.copy()

    classifiers = []
    votes = []
    #repeat T times...
    for i in range(T):
        print("starting iteration {} out of {}".format(i+1, T))
        #draw m examples uniformly with replacement
        bootDF = df.copy()
        bootDF.drop(bootDF.index,inplace=True) # creates copy and removes all rows
        
        for j in range (df.shape[0]): # gets random example from dataset
            randInt = random.randint(0, df.shape[0]-1)
            #bootDF._append(df.iloc[randInt])
            bootDF.loc[len(bootDF.index)] = df.iloc[randInt]
        
        #train a classifier
        print("classifier is being trained...")
        tree = ID3.id3(bootDF, label, 'Entropy', len(bootDF.columns)+1)
        classifiers.append(tree)
        error = 1 - evaluate(classifiers[i], _df, label)
        vote = 0.5 * math.log((1 - error) / error)
        print("the vote is: {}\n".format(vote))

        votes.append(vote)

    
    #construct final classifier by taking votes from all classifiers

    # return final hypothesis
    # hypothesis uses both votes and classifiers
    hypothesis = [votes, classifiers]

    return hypothesis
        
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
            return 1
        
# takes a hypothesis and tests it on a dataframe
def testHypothesis(hypothesis, df, label):
    # hypothesis is a list in the form [votes, classifiers]
    correctGuesses = 0
    prediction = 0
    temp = 0

    for index, row in df.iterrows():  # for each example
        for i in range(
            len(hypothesis[0])
        ):  # sums all iterations of the adaboost hypothesis
            guess = predict(hypothesis[1][i], row)
            temp += hypothesis[0][i] * guess

        prediction = numpy.sign(temp)

        if prediction == row[label]:
            correctGuesses += 1

    return correctGuesses / df.shape[0]


print("starting the bagging algorithm")
testDF = DataFormatting.finalBankTrainDF
hypothesis = baggedTrees(testDF, 'y', 5)
#open('BaggedOutput.txt', 'w').write(str(hypothesis))
print("done with bagging algorithm")
print("the hypothsis has accuracy: {}".format(testHypothesis(hypothesis, testDF, 'y')))