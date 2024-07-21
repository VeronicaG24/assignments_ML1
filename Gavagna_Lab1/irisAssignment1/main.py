# ----------------------------------------------------------------------------------------------------------------------
# Naive Bayes Classifier - Iris Dataset
# ----------------------------------------------------------------------------------------------------------------------
#
# Veronica Gavagna
#
# ----------------------------------------------------------------------------------------------------------------------
#
# Machine Learning for Robotics 1 - 2022/2023
# Assignment 1 pt.2:
# Think further - point 2:  If you have large data sets, you may incur in numerical errors while multiplying a lot of
#                           frequencies together. A solution is to work with log probabilities,
#                           by transforming logarithmically all probabilities and turning all multiplications into sums.
# Think further - point 3:  How would you proceed if the input variables were continuous?
#                           (Hint: variable values are used to compute probabilities by counting,
#                           but theory tells us that probabilities may be obtained by probability
#                           mass functions directly if they are known analytically.)
#                           You can experiment with the continuous variable case using
#                           the Iris data set (with its description). The four features can be made binary by
#                           (1) computing the average of each and (2) replacing each individual value with True or 1
#                           if it is above the mean and False or 0 if it is below.
#                           You can use the median in place of the average.
# ----------------------------------------------------------------------------------------------------------------------
#


import pandas as pd
import numpy as np
from sklearn import preprocessing
import NB_classifier


target_name = "Iris_Class"
path = "/Users/veronicagavagna/PycharmProjects/irisAssignment1/"


##############################################################
# Read a txt file, convert it into csv and create a dataframe
# RETURN:
#   Dataframe
##############################################################
def read_file_csv():
    read_file = pd.read_csv(path + 'iris.txt')
    read_file.to_csv(path + 'iris.csv', index=None)
    df = pd.read_csv(path + 'iris.csv', sep=',',
                     names=['Sepal_Length', 'Sepal_Width', 'Petal_Length', 'Petal_Width', 'Iris_Class'])
    df = pd.DataFrame(df, columns=['Sepal_Length', 'Sepal_Width', 'Petal_Length', 'Petal_Width', 'Iris_Class'])
    print("\n *************************** Original DataSet *************************** \n")
    print(df)
    return df


##############################################################
# Convert Dataset values into numeric values
# RETURN:
#   Dataset with numeric values
##############################################################
def ds_to_numeric(ds):
    data_set_numeric = ds.copy()
    le = preprocessing.LabelEncoder()
    data_set_numeric[:, targetIndex] = le.fit_transform(ds.T[targetIndex]) + 1
    return data_set_numeric


##############################################################
# Compute the average value for each level
# RETURN:
#   Dataset with binary values except for target column
##############################################################
def calculate_average_values():
    ds_average = np.zeros((1, dsNumeric.shape[1]), dtype=float)
    ds_average[0, :] = np.sum(dsNumeric, axis=0) / dataSet.shape[0]
    ds_average = np.delete(ds_average, targetIndex, axis=1)

    return ds_average


##############################################################
# Convert dataset to binary values by comparing with
# the average value
# RETURN:
#   Dataset with binary values except for target column
##############################################################
def convert_ds_to_binary(ds_numeric):
    ds_binary = np.full((ds_numeric.shape[0], ds_numeric.shape[1]-1), 9, dtype=int)
    ds_binary = np.insert(ds_binary, targetIndex, ds_numeric[:, targetIndex], axis=1)
    for i in range(0, ds_numeric.shape[0], 1):
        for j in range(0, ds_numeric.shape[1]-1, 1):
            if ds_numeric[i, j] >= dsAverage[0, j]:
                ds_binary[i, j] = 2
            else:
                ds_binary[i, j] = 1

    return ds_binary


##############################################################
# Create the test and training set by dividing the dataset
# numeric and print them
# RETURN:
#   Test set and training set with binary values
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
        for j in range(ds.shape[1]-1):
            if ds[i, j] > 2:
                ck = 1
                return ck
            else:
                ck = 0
        if ds[i, targetIndex] > 3:
            ck = 1
            return ck

    return ck


print("\n ++++++++++++++++++++++++++ Assignment 1 pt.2 ++++++++++++++++++++++++++ \n")
dataFrame = read_file_csv()
targetIndex = dataFrame.columns.get_loc(target_name)
print("\n ############## Original DataSet Matrix ############## \n")
dataSet = dataFrame.to_numpy()
if targetIndex != (dataSet.shape[1] - 1):
    dataSet[:, [targetIndex, (dataSet.shape[1] - 1)]] = dataSet[:, [(dataSet.shape[1] - 1), targetIndex]]
    print(" target moved to the last column of the dataset ")

print(dataSet)
dsNumeric = ds_to_numeric(dataSet)
print("\n ############## Numeric DataSet Matrix ############## \n")
print(dsNumeric)

print("############## Calculate average values ##############")
dsAverage = calculate_average_values()
print(dsAverage)

print("############## Dataset binary with average except for target column ##############")
dsBinary = convert_ds_to_binary(dsNumeric)
print(dsBinary)

print("############## Creation of Training and Test set binary ##############")
# 70% of values in training set, the remaining 30% of values in test set
trainingSet, testSet = create_train_test_set(70, dsBinary)
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

print("############## Naive Bayes Classifier ##############")
likelihoodProb, possibleValues, targetValues, testSetOK, targetProb = \
    NB_classifier.calculate_lh_probabilities(trainingSet, testSet, targetIndex)
testSet = testSetOK.copy()
testSet_old = testSet.copy()
print(testSet_old)
NB_classifier.test_NB_classifier(testSet, targetValues, likelihoodProb, possibleValues, targetIndex)
errorRate = NB_classifier.compute_error_rate(testSet_old, targetIndex, testSet)
