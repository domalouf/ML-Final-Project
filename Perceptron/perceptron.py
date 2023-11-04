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

xData = dataFormatting.trainX
yData = dataFormatting.trainY

print("trying out perceptron")
print(perceptron(xData, yData, 0.5, 10))