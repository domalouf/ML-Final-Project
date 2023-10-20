import pandas as pd
import numpy as np
import InfoGain

#attribute is the attribute that is added to tree and shrinks the dataset
def makeSubTree(attribute, df, label, labelValues, treeDepth, maxDepth):
    attributeValueCounts = df[attribute].value_counts(sort=False) #dictionary of the count of attribute values that are unique
    tree = {} #either the node or a sub tree

    for attributeValue, count in attributeValueCounts.items():
        attributeValueDF = df[df[attribute] == attributeValue] #dataset with rows of specific attribute

        pureNode = False #boolean for tracking if attributeValue is pure or not
        for v in labelValues:
            labelCount = attributeValueDF[attributeValueDF[label] == v].shape[0] #count of label v

            if labelCount == count: #all labels for this attribute are the same (it is pure)
                tree[attributeValue] = v #adding node to the tree
                df = df[df[attribute] != attributeValue] #removing rows with attributeValue
                pureNode = True

        if not pureNode:
          if treeDepth != maxDepth:
            tree[attributeValue] = "?" #branch is labeled with '?' because it isn't pure and needs expansion
          else: #if tree is at max depth, label node with most common value
            mostCommonVal = None
            mostCommonNum = 0

            for v in labelValues:
              commonCount = attributeValueDF[attributeValueDF[label] == v].shape[0] #count of label v
              if commonCount > mostCommonNum:
                mostCommonNum = commonCount
                mostCommonVal = v

            tree[attributeValue] = mostCommonVal

    return tree, df # tree is a dictionary with the tree node and its branches and df is updated dataset

# root is an empty dictionary that will contain the tree, pastAttributeValue will be the datatype of the previous attribute (will start as none and be recursively updated)
def makeTree(root, pastAttributeValue, df, label, labelValue, method, treeDepth, maxDepth):
    if df.shape[0] != 0: #if dataset still has elements after tree was updated
        maxInfoAttribute = InfoGain.bestAttribute(df, label, labelValue, method)
        tree, df = makeSubTree(maxInfoAttribute, df, label, labelValue, treeDepth, maxDepth) #getting tree and updated dataset
        nextRoot = None

        if pastAttributeValue != None: #add to next node of the tree
            root[pastAttributeValue] = dict()
            root[pastAttributeValue][maxInfoAttribute] = tree
            nextRoot = root[pastAttributeValue][maxInfoAttribute]
        else: #add tree root
            root[maxInfoAttribute] = tree
            nextRoot = root[maxInfoAttribute]

        if treeDepth < maxDepth:
          for node, branch in list(nextRoot.items()): #looks through nodes to check for non-pure nodes
              if branch == "?": #branch needs expansion
                  attributeValueData = df[df[maxInfoAttribute] == node] #using the updated dataset

                  makeTree(nextRoot, node, attributeValueData, label, labelValue, method, treeDepth + 1, maxDepth) #recursive call with updated dataset

def id3(df, label, method, maxTreeDepth):
    dfCopy = df.copy()
    tree = {}
    labelValues = dfCopy[label].unique()
    makeTree(tree, None, dfCopy, label, labelValues, method, 1, maxTreeDepth)
    return tree