import pandas as pd
import statistics

#Cars datasets formatting
carsTestDF = pd.read_csv ('DataSets/Car/test.csv', header=None, names=['buying', 'maint', 'doors', 'persons', 'lug-boot', 'safety', 'label'])
carsTrainDF = pd.read_csv ('DataSets/Car/train.csv', header=None, names=['buying', 'maint', 'doors', 'persons', 'lug-boot', 'safety', 'label'])


#Bank datasets formatting
bankColumns = ['age','job','marital','education','default','balance','housing',
               'loan','contact','day','month','duration','campaign','pdays','previous','poutcome','y']
bankTestDF = pd.read_csv ('DataSets/Bank/test.csv', header=None, names=bankColumns)
bankTrainDF = pd.read_csv ('DataSets/Bank/train.csv', header=None, names=bankColumns)
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


#now to replace the unknown values with the most common value in column 'poutcome'
bankTrainReplaceDF = bankTrainDFCopy.copy()
bankTestReplaceDF = bankTestDFCopy.copy()

#train dataset
poutcomeTrainList = bankTrainReplaceDF['poutcome'].tolist()
poutcomeTrainList = [x for x in poutcomeTrainList if x != 'unknown']
mostCommonTrainElement = statistics.mode(poutcomeTrainList)
bankTrainReplaceDF.loc[bankTrainReplaceDF['poutcome'] == 'unknown', 'poutcome'] = mostCommonTrainElement

#test dataset
poutcomeTestList = bankTestReplaceDF['poutcome'].tolist()
poutcomeTestList = [x for x in poutcomeTestList if x != 'unknown']
mostCommonTestElement = statistics.mode(poutcomeTestList)
bankTestReplaceDF.loc[bankTestReplaceDF['poutcome'] == 'unknown', 'poutcome'] = mostCommonTestElement

# dataframes to pass to prediction functions
finalCarTrainDF = carsTrainDF
finalCarTestDF = carsTestDF
finalBankTrainDF = bankTrainDFCopy
finalBankTestDF = bankTestDFCopy
finalBankTrainReplaceDF = bankTrainReplaceDF
finalBankTestReplaceDF = bankTestReplaceDF