import pandas as pd
import statistics

#Bank datasets formatting
bankColumns = ['age','job','marital','education','default','balance','housing',
               'loan','contact','day','month','duration','campaign','pdays','previous','poutcome','y']
bankTestDF = pd.read_csv ('EnsembleLearning/DataSets/Bank/test.csv', header=None, names=bankColumns)
bankTrainDF = pd.read_csv ('EnsembleLearning/DataSets/Bank/train.csv', header=None, names=bankColumns)
bankTestDFCopy = bankTestDF.copy()
bankTrainDFCopy = bankTrainDF.copy()

# the columns that are numeric and need to be converted to binary are age, balance, day, duration, campaign, pdays, and previous
numericColumns = ['age', 'balance', 'day', 'duration', 'campaign', 'previous'] #pdays not involved because of -1 value

#trying new cleaner method to convert numeric columns to binary
for column in numericColumns:
    bankTrainDFCopy.loc[bankTrainDFCopy[column] <= statistics.median(bankTrainDF[column].tolist()), column] = 0
    bankTrainDFCopy.loc[bankTrainDFCopy[column] > statistics.median(bankTrainDF[column].tolist()), column] = 1

#pdays uses reverse sorting order because it has special case of a negative median value
bankTrainDFCopy.loc[bankTrainDFCopy['pdays'] > statistics.median(bankTrainDF['pdays'].tolist()), 'pdays'] = 1
bankTrainDFCopy.loc[bankTrainDFCopy['pdays'] <= statistics.median(bankTrainDF['pdays'].tolist()), 'pdays'] = 0

#repeat for test data
for column in numericColumns:
    bankTestDFCopy.loc[bankTestDFCopy[column] <= statistics.median(bankTestDF[column].tolist()), column] = 0
    bankTestDFCopy.loc[bankTestDFCopy[column] > statistics.median(bankTestDF[column].tolist()), column] = 1

bankTestDFCopy.loc[bankTestDFCopy['pdays'] > statistics.median(bankTestDF['pdays'].tolist()), 'pdays'] = 1
bankTestDFCopy.loc[bankTestDFCopy['pdays'] <= statistics.median(bankTestDF['pdays'].tolist()), 'pdays'] = 0

#the label is labeled as a 1 or a -1 for adaBoost label label label
bankTrainDFCopy.loc[bankTrainDFCopy['y'] == 'yes', 'y'] = 1
bankTrainDFCopy.loc[bankTrainDFCopy['y'] == 'no', 'y'] = -1
bankTestDFCopy.loc[bankTestDFCopy['y'] == 'yes', 'y'] = 1
bankTestDFCopy.loc[bankTestDFCopy['y'] == 'no', 'y'] = -1

# dataframes to pass to prediction functions
finalBankTrainDF = bankTrainDFCopy.copy()
finalBankTestDF = bankTestDFCopy.copy()