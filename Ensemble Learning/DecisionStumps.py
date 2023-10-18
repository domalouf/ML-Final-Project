import InfoGain
import DataFormatting

# creates a decision tree with depth of 2
# the dataframe passed in must include weights for each example
# these weights will then be manipulated by the adaBoost algorithm
def makeDecisionStump(df, label):
    dfCopy = df.copy()
    tree = {}
    labelValues = dfCopy[label].unique()
    maxInfoAttribute = InfoGain.bestAttribute(df, label, labelValues, 'Entropy')
    attributeValueCounts = df[maxInfoAttribute].value_counts(sort=False) #dictionary of the count of attribute values that are unique
    tree = {} #either the node or a sub tree

    #loops through the different values an attribute can take and the amount of rows with that value
    for attributeValue, count in attributeValueCounts.items():
        attributeValueDF = df[df[maxInfoAttribute] == attributeValue] #dataset with rows of specific attribute
        totalAttributeWeight = attributeValueDF['weight'].sum()

        pureNode = False #boolean for tracking if attributeValue is pure or not
        for v in labelValues:

            #adds the total weights of the rows with the specific attribute and label values
            labelWeights = attributeValueDF[attributeValueDF[label] == v]['weight'].sum()

            if labelWeights == totalAttributeWeight: #all labels for this attribute are the same (it is pure)
                tree[attributeValue] = v #adding node to the tree
                df = df[df[maxInfoAttribute] != attributeValue] #removing rows with attributeValue
                pureNode = True

        if not pureNode:
            highestWeightValue = None
            highestWeightNum = 0

            for v in labelValues:
              weightCount = attributeValueDF[attributeValueDF[label] == v]['weight'].sum()
              if weightCount > highestWeightNum:
                highestWeightNum = weightCount
                highestWeightValue = v

            tree[attributeValue] = highestWeightValue
    return tree