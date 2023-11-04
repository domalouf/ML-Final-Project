import csv

with open('Perceptron/DataSets/train.csv', newline='') as csvfile:
    train = list(csv.reader(csvfile))

with open('Perceptron/DataSets/test.csv', newline='') as csvfile:
    test = list(csv.reader(csvfile))

# Separate the nx5 array into a nx4 array and a nx1 array
trainX = [row[:4] for row in train]  # Extract the first 4 columns of each row
trainY = [row[4] for row in train]   # Extract the last column of each row

testX = [row[:4] for row in test]  # Extract the first 4 columns of each row
testY = [row[4] for row in test]   # Extract the last column of each row

# Convert strings to doubles and ints
trainX = [[float(s) for s in row] for row in trainX]
trainY = [int(s) for s in trainY]
testX = [[float(s) for s in row] for row in testX]
testY = [int(s) for s in testY]

# Replace 0 with -1 for y data
for i in range(len(trainY)):
    if trainY[i] == 0:
        trainY[i] = -1

for i in range(len(testY)):
    if testY[i] == 0:
        testY[i] = -1