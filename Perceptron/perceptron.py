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

xData = dataFormatting.trainX
yData = dataFormatting.trainY

print("trying out perceptron")
print(perceptron(xData, yData, 0.5, 10))

print("trying out voted perceptron")
#print(votedPerceptron(xData, yData, 0.5, 10)) # prints a lot of things

print("trying out average perceptron")
print(averagePerceptron(xData, yData, 0.5, 10))
