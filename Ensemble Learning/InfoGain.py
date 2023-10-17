import pandas as pd
import numpy as np

# df is a data frame with a specific value for an attribute (e.g. only data with Outlook = Sunny)
# label is attribute that is the predictor and labelValues is list of possible values (e.g. ["yes","no"])
def totalEntropy(df, label, labelValues):
    numRows = df.shape[0] #number of rows in subset of dataframe that is passed to function
    entropy = 0

    for v in labelValues:
        labelCount = df[df[label] == v].shape[0] #rows with label value v
        labelEntropy = 0
        if labelCount != 0:
            labelProb = labelCount/numRows #label probability
            labelEntropy = - labelProb * np.log2(labelProb)  #labelEntropy
        entropy += labelEntropy
    return entropy

# Majority Error
def totalME(dataSet, label, labelValues):
    numRows = dataSet.shape[0]
    commonLabelCount = 0

    for v in labelValues:
        labelCount = dataSet[dataSet[label] == v].shape[0] #rows with label value v
        if labelCount > commonLabelCount:
          commonLabelCount = labelCount

    ME = (numRows - commonLabelCount)/numRows
    return ME

#dfSubset is a data frame with a specific value for an attribute (e.g. only data with Outlook = Sunny), label and labelValues are same as above function
def totalGI(df, label, labelValues):
    numRows = df.shape[0]
    GI = 0

    for v in labelValues:
        labelCount = df[df[label] == v].shape[0] #rows with label value v
        labelGI = 0
        if labelCount != 0:
            labelProb = labelCount/numRows
            labelGI = 1 - (labelProb)**2 - (1 - labelProb)**2

    GI = labelGI
    return GI

# attribute is the name of the attribute we want information gain from (e.g. "outlook"), df is data frame, label and labelValues are same as above
def informationGain(attribute, df, label, labelValues, method):
    attributeValues = df[attribute].unique()
    numRows = df.shape[0]
    attributeInfoGain = 0.0

    for av in attributeValues:
        attributeValueData = df[df[attribute] == av] #rows with that attribute
        attributeValueCount = attributeValueData.shape[0]
        if method == 'Entropy':
          attributeValueEntropy = totalEntropy(attributeValueData, label, labelValues)
        if method == 'ME':
          attributeValueEntropy = totalME(attributeValueData, label, labelValues)
        if method == 'GI':
          attributeValueEntropy = totalGI(attributeValueData, label, labelValues)
        attributeValueProbability = attributeValueCount/numRows
        attributeInfoGain += attributeValueProbability * attributeValueEntropy

    return totalEntropy(df, label, labelValues) - attributeInfoGain

#df is entire dataframe, label is attribute acting as label, labelValues is list of values for that label
def bestAttribute(df, label, labelValues, method):
    attributes = df.columns.drop(label) #gets names of attributes minus the label

    maxGain = -1
    maxAttribute = None

    for f in attributes:
        attributeGain = informationGain(f, df, label, labelValues, method)
        if maxGain < attributeGain:
            maxGain = attributeGain
            maxAttribute = f

    return maxAttribute