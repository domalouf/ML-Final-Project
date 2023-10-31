import sys
#sys.path.append("..\\ML-Final-Project")
sys.path.append("..\\ML-Final-Project\\DecisionTree")
import InfoGain
import ID3
import Predictions

import DataFormatting

trainingDFMedian = DataFormatting.finaltrainDFMedian
testingDFMedian = DataFormatting.finaltestDFMedian
trainingDFMean = DataFormatting.finaltrainDFMean
testingDFMean = DataFormatting.finaltestDFMean

testingDFMedian.drop(columns=['ID'])
testingDFMean.drop(columns=['ID'])

# print("making tree1")
# treeMedian = ID3.id3(trainingDFMedian, 'income>50K', 'Entropy', len(trainingDFMedian.columns)+1)
# print("tree1 is done")

# print("making tree2")
# treeMean = ID3.id3(trainingDFMean, 'income>50K', 'Entropy', len(trainingDFMean.columns)+1)
# print("tree2 is done")

print("making tree3")
treeMedianME = ID3.id3(trainingDFMean, 'income>50K', 'ME', len(trainingDFMean.columns)+1)
print("tree2 is done")

print("making tree4")
treeMedianGI = ID3.id3(trainingDFMean, 'income>50K', 'GI', len(trainingDFMean.columns)+1)
print("tree2 is done")


def kagglePredict(tree, df, f):
    for index, row in df.iterrows(): #loops through rows in dataset
        result = Predictions.predict(tree, df.iloc[index]) #predict the row
        if result == None:
            result = 0
        f.write("{},{}\n".format(index+1, result))


# print("writing to file1 ...")
# f = open("KaggleCompetition/Submissions/submission1.csv", "w")
# f.write("ID,Prediction\n")
# kagglePredict(treeMedian, testingDFMedian, f)
# f.close()

# print("writing to file2 ...")
# f = open("KaggleCompetition/Submissions/submission2.csv", "w")
# f.write("ID,Prediction\n")
# kagglePredict(treeMean, testingDFMean, f)
# f.close()

print("writing to file3 ...")
f = open("KaggleCompetition/Submissions/submission3.csv", "w")
f.write("ID,Prediction\n")
kagglePredict(treeMedianME, testingDFMedian, f)
f.close()

print("writing to file4 ...")
f = open("KaggleCompetition/Submissions/submission4.csv", "w")
f.write("ID,Prediction\n")
kagglePredict(treeMedianGI, testingDFMedian, f)
f.close()

print("donezo")