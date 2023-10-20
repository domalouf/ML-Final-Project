import pandas as pd
import numpy as np


# df is a data frame with a specific value for an attribute (e.g. only data with Outlook = Sunny)
# label is attribute that is the predictor and labelValues is list of possible values (e.g. ["yes","no"])
def totalEntropy(df, label, labelValues):
    entropy = 0
    totalWeight = df["adaWeight"].sum()

    for v in labelValues:
        labelWeight = df[df[label] == v]["adaWeight"].sum()
        labelEntropy = 0
        if labelWeight != 0:
            labelProb = labelWeight / totalWeight  # label probability
            labelEntropy = -labelProb * np.log2(labelProb)  # labelEntropy
        entropy += labelEntropy
    return entropy


# attribute is the name of the attribute we want information gain from (e.g. "outlook"), df is data frame, label and labelValues are same as above
def informationGain(attribute, df, label, labelValues):
    attributeValues = df[attribute].unique()
    attributeInfoGain = 0.0

    for av in attributeValues:
        attributeValueData = df[df[attribute] == av]  # rows with that attribute value
        attributeValueWeight = attributeValueData["adaWeight"].sum()
        attributeValueEntropy = totalEntropy(attributeValueData, label, labelValues)
        attributeValueProbability = attributeValueWeight / df["adaWeight"].sum()
        attributeInfoGain += attributeValueProbability * attributeValueEntropy

    return totalEntropy(df, label, labelValues) - attributeInfoGain


# df is entire dataframe, label is attribute acting as label, labelValues is list of values for that label
def bestAttribute(df, label, labelValues):
    attributes = df.columns.drop(label).drop(
        "adaWeight"
    )  # gets names of attributes minus the label and weights

    maxGain = -1
    maxAttribute = None

    for f in attributes:
        attributeGain = informationGain(f, df, label, labelValues)
        if attributeGain > maxGain:
            maxGain = attributeGain
            maxAttribute = f

    return maxAttribute
