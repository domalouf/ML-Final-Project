import numpy as np
import random
import dataFormatting

def perceptron(i_x, i_y, r, T):
    _x = i_x
    _y = i_y

    # initialize w as 0's
    w = [0] * len(_x[0])

    for e in range(T):
        #shuffle data
        row_indices = list(range(len(_x)))
        random.shuffle(row_indices)
        shuffled_x = [_x[i] for i in row_indices]
        shuffled_y = [_y[i] for i in row_indices]

        #for each example xy
        for index, xRow in enumerate(shuffled_x):

            #if wrong prediction - update w
            if np.sign(np.dot(xRow,w)) * shuffled_y[index] <= 0:
                temp = [x * shuffled_y[index] for x in xRow] # multiplies x by -1 or 1
                temp2 = [z * r for z in temp] # factors in r
                w = [a + b for a, b in zip(w, temp2)] # updates w

    return w

def votedPerceptron(i_x, i_y, r, T):
    _x = i_x
    _y = i_y

    # initialize w as list
    w = []
    # first element is all 0's
    w.append([0] * len(_x[0]))

    #initialize m as 0
    m = 0
    #initialize c as a list
    c = []
    c.append(0)

    for e in range(T):
        #for each example xy
        for index, xRow in enumerate(_x):

            #if wrong prediction - update w
            if np.sign(np.dot(xRow,w[m])) * _y[index] <= 0:
                temp = [x * _y[index] for x in xRow] # multiplies x by -1 or 1
                temp2 = [z * r for z in temp] # factors in r
                w.append([a + b for a, b in zip(w[m], temp2)]) # adds new w to list of w's
                m = m + 1 # update m
                c.append(1)

            # if correct prediction - add one to c
            else:
                c[m] = c[m] + 1

    return combine_matrix_rows_with_list(w, c)

def averagePerceptron(i_x, i_y, r, T):
    _x = i_x
    _y = i_y

    # initialize w and a as 0's
    w = [0] * len(_x[0])
    a = [0] * len(_x[0])

    for e in range(T):
        #shuffle data
        row_indices = list(range(len(_x)))
        random.shuffle(row_indices)
        shuffled_x = [_x[i] for i in row_indices]
        shuffled_y = [_y[i] for i in row_indices]

        #for each example xy
        for index, xRow in enumerate(shuffled_x):

            #if wrong prediction - update w
            if np.sign(np.dot(xRow,w)) * shuffled_y[index] <= 0:
                temp = [x * shuffled_y[index] for x in xRow] # multiplies x by -1 or 1
                temp2 = [z * r for z in temp] # factors in r
                w = [a + b for a, b in zip(w, temp2)] # updates w
            # update a with w
            a = [z + b for z, b in zip(a, w)]

    return a

# helper function for the return value of votedPerceptron
def combine_matrix_rows_with_list(matrix, x):
    combined_list = []
    for i, row in enumerate(matrix):
        if i < len(x):
            combined_list.append((row, x[i]))
        else:
            combined_list.append((row, None))  # If there are not enough elements in x, use None for the missing value.
    return combined_list

# used for both perceptron and average perceptron
# x and y are for test data, w is weight vector from perceptron
def evaluatePerceptron(x, y, w):
    wrongCount = 0

    #for each example xy
    for index, xRow in enumerate(x):

        #if wrong prediction - add 1 to wrong count
        if np.sign(np.dot(xRow,w)) * y[index] <= 0:
            wrongCount = wrongCount + 1

    # returns error of weight vector
    return wrongCount/len(y)

# x and y are for test data, vp is voted perceptron values for weight vectors and counts
# vp is in form: [([weight vector], count), (([weight vector], count))]
def evaluateVotedPerceptron(x, y, vp):
    wrongCount = 0

    #for each example xy
    for index, xRow in enumerate(x):
        prediction = 0

        # for each weight vector and count
        for weight in vp:

            # adds up prediction signs
            prediction = prediction + weight[1] * np.sign(np.dot(xRow,weight[0]))

        #if wrong prediction - add 1 to wrong count
        if prediction * y[index] <= 0:
            wrongCount = wrongCount + 1

    # returns error of weight vector
    return wrongCount/len(y)


xData = dataFormatting.trainX
yData = dataFormatting.trainY

xTest = dataFormatting.testX
yTest = dataFormatting.testY

learningRate = 0.5
numEpochs = 10

print("all calls use learning rate of 0.5 and T of 10\n")
print("Running normal perceptron 100 times and getting average error...")
averageError = 0
for i in range(100):
    normalWeights = perceptron(xData, yData, learningRate, numEpochs)
    error = evaluatePerceptron(xTest, yTest, normalWeights)
    averageError = averageError + error

averageError = round(averageError/100, 5)
print("the average error is: {}\n".format(averageError))

print("Running voted perceptron 20 times and getting average error...")
averageError = 0
for i in range(20):
    votedWeights = votedPerceptron(xData, yData, learningRate, numEpochs)
    error = evaluateVotedPerceptron(xTest, yTest, votedWeights)
    averageError = averageError + error

averageError = round(averageError/20, 5)
print("the average error is: {}\n".format(averageError))

print("Running average perceptron 100 times and getting average error...")
averageError = 0
for i in range(100):
    averageWeights = averagePerceptron(xData, yData, learningRate, numEpochs)
    error = evaluatePerceptron(xTest, yTest, averageWeights)
    averageError = averageError + error

averageError = round(averageError/100, 5)
print("the average error is: {}\n".format(averageError))

print("the weight vector from one call to perceptron is: {}\n".format(perceptron(xData, yData, learningRate, numEpochs)))
print("the weight vector from one call to averagePerceptron is: {}\n".format(averagePerceptron(xData, yData, learningRate, numEpochs)))
print("the weight vector and counts from one call to votedPerceptron is:\n")

votedValues = votedPerceptron(xData, yData, learningRate, numEpochs)

# rounds values to 8 digits for better looking output
votedValues = [([round(value, 8) for value in tup[0]], tup[1]) for tup in votedValues]
print(votedValues)

f = open("Perceptron/votedOutput.txt", "w")
f.write(str(votedValues))
f.close()

print("the voted perception output is also contained in the file votedOutput.txt")