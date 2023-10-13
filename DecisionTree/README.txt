How to create a decision tree:
Call the ID3 function in ID3.py
ID3(df, label, method, maxTreeDepth)
df: a pandas data frame of the dataset that the decision tree will represent
label: a string for the attribute in the dataset that is being predicted by the tree
method: a string for 'Entropy', 'ME', or 'GI' (the heuristic that the tree will split by)
maxTreeDepth: an integer for how many levels the tree will grow

Hello Grader!

I will list the question numbers for HW1 and where to find the code that answers that question

Question 2
a. The ID3 algorithm is in ID3.py with helper functions in InfoGain.py
b. When CarsDataOutput.py is run, two tables are printed that correspond to the train and test data prediction accuracy (takes ~ 1 minute)

Question 3
a. BankDataOutput.py will print the table for bank train prediction and bank test prediction (takes ~20 minutes)
b. BDOReplacement.py will print the tables for the bank train and tests with replacement (also takes ~20 minutes)
Both of these files will print some progress report to show what part of the table the function is working on

run.sh will run CarsDataOutput.py, BankDataOutput.py, and BDOReplacement.py in that order

Thanks for grading!

For all my non-grading homies looking to understand this folder
ID3.py creates a decision tree of a dataset,
InfoGain.py has functions for finding entropy, majority error, and gini index,
Prediction.py compares tree predictions to a test dataset,
DataFormatting.py, BDOReplacement.py, BankDataOutput.py, and CarsDataOutput.py are all specific to the assignment
