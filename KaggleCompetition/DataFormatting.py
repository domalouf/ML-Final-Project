import pandas as pd
import statistics

#datasets formatting
trainDataColumns = ['age','workclass','fnlwgt','education','education.num','marital.status','occupation','relationship','race','sex','capital.gain','capital.loss','hours.per.week','native.country','income>50K']
testDataColumns = ['ID','age','workclass','fnlwgt','education','education.num','marital.status','occupation','relationship','race','sex','capital.gain','capital.loss','hours.per.week','native.country']

trainDF = pd.read_csv ('KaggleCompetition/DataSets/train_final.csv', header=None, names=trainDataColumns)
testDF = pd.read_csv ('KaggleCompetition/DataSets/test_final.csv', header=None, names=testDataColumns)
trainDFCopy1 = trainDF.copy()
testDFCopy1 = testDF.copy()
trainDFCopy2 = trainDF.copy()
testDFCopy2 = testDF.copy()

# columns that are numeric and need to be converted to binary
numericColumns = ['age', 'fnlwgt', 'education.num', 'capital.gain', 'capital.loss', 'hours.per.week']

#convert numeric columns to binary by median
for column in numericColumns:
    trainDFCopy1.loc[trainDFCopy1[column] <= statistics.median(trainDF[column].tolist()), column] = 0
    trainDFCopy1.loc[trainDFCopy1[column] > statistics.median(trainDF[column].tolist()), column] = 1

#repeat for test data
for column in numericColumns:
    testDFCopy1.loc[testDFCopy1[column] <= statistics.median(testDF[column].tolist()), column] = 0
    testDFCopy1.loc[testDFCopy1[column] > statistics.median(testDF[column].tolist()), column] = 1

#convert numeric columns to binary by mean
for column in numericColumns:
    trainDFCopy2.loc[trainDFCopy2[column] <= statistics.mean(trainDF[column].tolist()), column] = 0
    trainDFCopy2.loc[trainDFCopy2[column] > statistics.mean(trainDF[column].tolist()), column] = 1

#repeat for test data
for column in numericColumns:
    testDFCopy2.loc[testDFCopy2[column] <= statistics.mean(testDF[column].tolist()), column] = 0
    testDFCopy2.loc[testDFCopy2[column] > statistics.mean(testDF[column].tolist()), column] = 1

# dataframes to pass to prediction functions
finaltrainDFMedian = trainDFCopy1.copy()
finaltestDFMedian = testDFCopy1.copy()
finaltrainDFMean = trainDFCopy2.copy()
finaltestDFMean = testDFCopy2.copy()