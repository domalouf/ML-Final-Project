#This is the Ensemble Learning folder

The functions here don't work as intended, so I wouldn't recommend trying them out

To use Ada Boost, go to AdaBoost.py and call adaBoost function as follows

adaBoost(df, label, T)
df is a pandas data frame, label is the attribute to be predicted, and T is the number of iterations

To use Bagged Trees, go to BaggedTrees.py and call baggedTrees function

baggedTrees(df, label, T)
df is a pandas data frame, label is the attribute to be predicted, and T is the number of iterations

To use Random Forest, go to RandomForest.py and call randomForest function

randomForest(df, label, T, G)
df is a pandas data frame, label is the attribute to be predicted, T is the number of iterations, and G is the number of attributes to be sampled
