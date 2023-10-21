import DecisionStumps
import DataFormatting
import math
import numpy


# df is a dataframe with the label values as 1 and -1
# T is the number of times the algorithm is repeated
def adaBoost(_df, label, T):
    df = _df.copy()
    # initialize the weights as uniform

    df["adaWeight"] = 1 / df.shape[0]  # all weights are 1/total number of examples

    classifiers = []  # list of classifiers
    votes = []  # list of votes

    # repeat for T times
    for i in range(T):
        print("performing iteration {} out of {}".format(i+1, T))
        # 1) find classifier
        classifiers.append(DecisionStumps.makeDecisionStump(df, label))

        # 2) compute its vote
        # the error is the sum of the weights when the classifier is incorrect
        error = evaluateError(classifiers[i], df, label)
        vote = 0.5 * math.log((1 - error) / error)

        votes.append(vote)

        # 3) update the values of the weights of training examples

        for index, row in df.iterrows():  # loops through rows in dataset
            if (
                predict(classifiers[i], row) == row[label]
            ):  # compares prediction to true value
                prediction = 1
            else:
                prediction = -1

            df.at[index, "adaWeight"] *= math.exp(
                -vote * prediction
            )  # updates weight based on prediction scores

        df['adaWeight'] /= df['adaWeight'].sum() # normalizes weights to sum up to 1

    # return final hypothesis
    # hypothesis uses both votes and classifiers
    hypothesis = [votes, classifiers]

    return hypothesis


# returns accuracy of a decision tree tested on a dataframe
def evaluateError(tree, df, label):
    wrongWeight = 0
    for index, row in df.iterrows():  # loops through examples
        result = predict(tree, row)  # predict the example
        # if the prediction is wrong
        if result != row[label]:
            wrongWeight += row["adaWeight"]
    return wrongWeight


# instance is a list of values for attributes that will be passed into the tree (e.g. [‘Outlook’][‘Rain’][‘Wind’][‘Strong’])
def predict(tree, instance):
    attribute = next(iter(tree))
    return tree[attribute][instance[attribute]] # gets the value of the stump from the instance's value


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


testDF = DataFormatting.finalBankTrainDF

print("starting adaBoost call")
tempHypothesis = adaBoost(testDF, "y", 10)
print("done with adaBoost call")
print("this is the accuracy for adaBoost on the dataset it was trained on: {}".format(testHypothesis(tempHypothesis, testDF, "y")))
