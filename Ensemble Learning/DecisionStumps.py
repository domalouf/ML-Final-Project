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

    
    for attributeValue, count in attributeValueCounts.items():
        attributeValueDF = df[df[maxInfoAttribute] == attributeValue] #dataset with rows of specific attribute

        pureNode = False #boolean for tracking if attributeValue is pure or not
        for v in labelValues:
            #adds the total weights of the rows with the attribute
            labelCount = attributeValueDF[attributeValueDF[label] == v].shape[0] #count of label v

            if labelCount == count: #all labels for this attribute are the same (it is pure)
                tree[attributeValue] = v #adding node to the tree
                df = df[df[maxInfoAttribute] != attributeValue] #removing rows with attributeValue
                pureNode = True

        if not pureNode:
            mostCommonVal = None
            mostCommonNum = 0

            for v in labelValues:
              commonCount = attributeValueDF[attributeValueDF[label] == v].shape[0] #count of label v
              if commonCount > mostCommonNum:
                mostCommonNum = commonCount
                mostCommonVal = v

            tree[attributeValue] = mostCommonVal
    return tree