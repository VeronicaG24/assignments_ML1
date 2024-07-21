# ----------------------------------------------------------------------------------------------------------------------
# Naive Bayes Classifier - Weather Dataset
# ----------------------------------------------------------------------------------------------------------------------
#
# Veronica Gavagna
#
# ----------------------------------------------------------------------------------------------------------------------
#
# Machine Learning for Robotics 1 - 2022/2023
# Assignment 1 pt.1:
# Task 1: Data preprocessing
# Task 2: Build a naive Bayes classifier
# Task 3: Improve the classifier with Laplace (additive) smoothing
# Think further - point 1: If you have large data sets, you may incur in numerical errors while multiplying a lot of
#                          frequencies together. A solution is to work with log probabilities,
#                          by transforming logarithmically all probabilities and turning all multiplications into sums.
# ----------------------------------------------------------------------------------------------------------------------
#


import pandas as pd
import numpy as np
from sklearn import preprocessing
import NB_classifier
import NB_Logarithmic
import NB_LaplaceSmoothing

target_name = "Play"
path = "/Users/veronicagavagna//Documents/PycharmProjects/assignment1_ML1/"


##############################################################
# Read a txt file, convert it into csv and create a dataframe
# RETURN:
#   Dataframe
##############################################################
def read_file_csv():
    read_file = pd.read_csv(path + 'weather.txt')
    read_file.to_csv(path + 'weather.csv', index=None)
    df = pd.read_csv(path + 'weather.csv', delim_whitespace=True)
    print("\n *************************** Original DataSet *************************** \n")
    print(df)
    return df


##############################################################
# Convert Dataset values into numeric values
# RETURN:
#   Dataset with numeric values
##############################################################
def ds_to_numeric(ds):
    data_set_numeric = np.zeros((ds.shape[0], ds.shape[1]), dtype=int)
    le = preprocessing.LabelEncoder()
    for j in range(ds.shape[1]):
        data_set_numeric[:, j] = le.fit_transform(ds.T[j]) + 1
    return data_set_numeric


##############################################################
# Create the test and training set by dividing the dataset
# numeric and print them
# RETURN:
#   Test set and training set with numeric values
##############################################################
def create_train_test_set(percentage, ds):
    np.random.shuffle(ds)
    end_training = int(ds.shape[0]*percentage/100)
    end_test = ds.shape[0]
    training_set = ds[:end_training]
    test_set = ds[end_training:end_test]
    print("\n *************************** Training Set *************************** \n")
    print(training_set)
    print("\n *************************** Test Set *************************** \n")
    print(test_set)
    return training_set, test_set


##############################################################
# Check that values in a dataset are valid
# RETURN:
#   0 if all values are valid, 1 if not
##############################################################
def check_dataSet(ds):
    ck = 0
    for i in range(ds.shape[0]):
        for j in range(ds.shape[1]):
            if ds[i, j] < 1:
                ck = 1
                return ck
            else:
                ck = 0
    return ck


print("\n ++++++++++++++++++++++++++ Assignment 1 pt.1 ++++++++++++++++++++++++++ \n")
dataFrame = read_file_csv()
targetIndex = dataFrame.columns.get_loc(target_name)
print("\n ############## Original DataSet Matrix ############## \n")
dataSet = dataFrame.to_numpy()
print(dataSet)

if targetIndex != (dataSet.shape[1] - 1):
    dataSet[:, [targetIndex, (dataSet.shape[1] - 1)]] = dataSet[:, [(dataSet.shape[1] - 1), targetIndex]]
    print(" target moved to the last column of the dataset ")

dsNumeric = ds_to_numeric(dataSet)
print("\n ############## Numeric DataSet Matrix ############## \n")
print(dsNumeric)

print("############## Creation of Training and Test set binary ##############")
# 10 values in training set, the remaining 4 values in test set
trainingSet, testSet = create_train_test_set(71, dsNumeric)
checkTraining = check_dataSet(trainingSet)
checkTest = check_dataSet(testSet)

if checkTraining == 1:
    print("\n *************** training set not valid *************** \n")
    raise Exception("Sorry, training set not valid")
else:
    print("\n *************** training set is valid *************** \n")

if checkTest == 1 and (testSet.shape[1] != trainingSet.shape[1] or testSet.shape[1] != trainingSet.shape[1] - 1):
    print("\n *************** test set not valid *************** \n")
    raise Exception("Sorry, test set not valid")
else:
    if testSet.shape[1] == (trainingSet.shape[1] - 1):
        testSet = np.insert(testSet, targetIndex, np.zeros(testSet.shape[0]), axis=1)
        print("\n *************** test set has now d+1 columns *************** \n")
        print(testSet)
    print("\n *************** test set is valid *************** \n")

print("\n ############## Naive Bayes Classifier ############## \n")

likelihoodProb, possibleValues, targetValues, testSetOK, targetProb = \
    NB_classifier.calculate_lh_probabilities(trainingSet, testSet, targetIndex)
testSet = testSetOK.copy()
testSet_old = testSet.copy()
print(testSet_old)
NB_classifier.test_NB_classifier(testSet, targetValues, likelihoodProb, possibleValues, targetIndex)
errorRate = NB_classifier.compute_error_rate(testSet_old, targetIndex, testSet)

print("\n ############## Laplace Smoothing ############## \n")
num_levels = np.array([3, 3, 2, 2, 2])
a = 0.01  #test with: 0,1 - 0,01 ...

trainingSetL = trainingSet.copy()
testSet_oldL = testSet_old.copy()
trainingSetL = np.insert(trainingSetL, 0, num_levels, axis=0)
testSetL = np.insert(testSet_oldL, 0, num_levels, axis=0)
print("*************** Training Test for Laplace Smoothing *************** \n")
print(trainingSetL)
print("*************** Test Test for Laplace Smoothing *************** \n")
print(testSetL)

likelihoodProbL, possibleValuesL, targetValuesL, testSetOKL, probTargetL = \
    NB_LaplaceSmoothing.calculate_lh_probabilitiesL(trainingSetL, targetIndex, a, testSetL, num_levels)
testSetL = testSetOKL.copy()
testSet_oldL = testSetL.copy()
print(testSet_oldL)
NB_LaplaceSmoothing.test_NB_classifierL(
    testSetL, targetValuesL, possibleValuesL, likelihoodProbL, targetIndex)
errorRateL = NB_LaplaceSmoothing.compute_error_rateL(testSet_oldL, targetIndex, testSetL)

print("\n ############## Logarithmic Transformation only with large data sets ############## \n")
if dataSet.shape[0] > 5000:
    log_likelihoodProb, log_possibleValues, log_targetValues, log_testSetOK, log_targetProb = \
        NB_Logarithmic.log_calculate_lh_probabilities(trainingSet, targetIndex, testSet)
    log_testSet = log_testSetOK.copy()
    log_testSet_old = log_testSet.copy()
    print(log_testSet_old)
    NB_Logarithmic.log_test_NB_classifier(
        log_testSet, log_targetValues, log_possibleValues, log_likelihoodProb, targetIndex)
    log_errorRate = NB_Logarithmic.log_compute_error_rate(log_testSet_old, targetIndex, log_testSet)
