# ----------------------------------------------------------------------------------------------------------------------
# kNN Classifier
# ----------------------------------------------------------------------------------------------------------------------
#
# Veronica Gavagna
#
# ----------------------------------------------------------------------------------------------------------------------
#
# Machine Learning for Robotics 1 - 2022/2023
# Assignment 3:
# Task 1: Obtain a data set
# Task 2: Build a kNN classifier
#       2.1: Create a program that takes the following parameters as input arguments:
#            1. a set of data, as an n x d matrix, to be used as the training set
#            2. a corresponding column (n x 1 matrix) of targets, i.e., class labels
#            3. another set of data, as an m x d matrix, to be used as the test set an integer k
#            4. OPTIONALLY, a set of data, as an m x 1 matrix, to be used as the test set ground truth (class labels)
#       2.2: The program should:
#            1. Check that the number of arguments received equals at least the number of mandatory arguments
#            2. Check that the number of columns of the second matrix equals the number of columns of the first matrix
#            3. Check that k>0 and k<=cardinality of the training set (number of rows, above referred to as n)
#            4. Classify the test set according to the kNN rule, and return the classification obtained
#            5. If the test set has the optional additional column, use this as a target:
#               compute and return the error rate obtained (number of errors / m)
# Task 3: Test the kNN classifier
#       3.1: Compute the accuracy on the test set: on 10 tasks, i.e., each digit vs the remaining 9
#       3.2: Compute the accuracy on the test set: for several values of k, e.g., k=1,2,3,4,5,10,15,20,30,40,50
#       3.2: Provide data or graphs for any combination of these parameters
#       ADDENDUM:
#       Subsample 10 times the whole training set by taking a random 10% of the data (6K instances).
#       For each of the experiments above, compute a confusion matrix, the classification quality indexes.
#       Provide an indication of typical value (average) and an appropriate measure of spread (standard deviation).
#       Summarise these in appropriate tables.
#
# ----------------------------------------------------------------------------------------------------------------------
#


import warnings
import random
import numpy as np
import matplotlib.pyplot as plt
import os
import knnClassifier as kNN
import createCSV
import math

current_directory = os.getcwd()
path_directory = "/Users/veronicagavagna/PycharmProjects/Gavagna-Lab3"
colors = list(['red', 'blue', 'orange', 'magenta', 'yellow', 'green', 'purple', 'cyan', 'brown', 'pink'])
numSubsets = int(10)
percentageSubsets = int(10)
numDigits = int(10)


##############################################################
# Read mnist file and obtain the dataset divided into
# training and test set
# RETURN:
#   training set and its labels, test set and its labels
##############################################################
def import_data_set():
    warnings.simplefilter("ignore")
    from tensorflow.keras.datasets import mnist
    (train_x, train_y), (test_x, test_y) = mnist.load_data(path=current_directory + '/mnist.npz')
    print("MNIST Dataset Shape:")
    print("train_X: " + str(train_x.shape))
    print("train_Y: " + str(train_y.shape))
    print("test_X: " + str(test_x.shape))
    print("test_Y: " + str(test_y.shape))
    return train_x, train_y, test_x, test_y


##############################################################
# Create numSub random subsets
# RETURN:
#   random training set and their associated labels
##############################################################
def get_random_subsets(trainXmat, trainYlist, percentage, numSub):
    randTrainX = np.zeros((numSub, int(trainXmat.shape[0] / percentage), trainXmat.shape[1]))
    randTrainY = np.zeros((numSub, int(trainXmat.shape[0] / percentage)))
    for i in range(0, numSub, 1):
        randomIndexList = random.sample(range(0, trainXmat.shape[0] - 1), int(trainXmat.shape[0] / percentage))
        for j in range(0, int(trainXmat.shape[0] / percentage), 1):
            for s in range(0, 28 * 28, 1):
                randTrainX[i, j, s] = trainXmat[int(randomIndexList[j]), s]
                randTrainY[i, j] = trainYlist[int(randomIndexList[j])]

    return randTrainX, randTrainY


############################################################
#
# Graph in a bar chart the accuracy for binary comparison
# Save the .png image
#
############################################################
def graph_accuracyBinary(acc, k_val):
    x_pos = np.arange(len(k_val))
    for c in range(0, numDigits, 1):
        graph = plt.bar(x_pos, acc[c, :], align='center', color=colors[c])
        plt.xticks(x_pos, k_val)
        plt.ylabel("Accuracy")
        plt.xlabel("K values")
        for p in graph:
            width = p.get_width()
            height = p.get_height()
            x, y = p.get_xy()
            plt.text(x + width / 2, y + height * 1.01, f'{height:.2f}' + '%', ha='center', weight='ultralight',
                     size='x-small')

        plt.title("Accuracy for recognizing " + str(c) + "vs. remaining class")
        plt.savefig(str(c) + "vs.Other.png")
        plt.show()


############################################################
#
# Graph in a bar chart the accuracy for not-binary
# Save the .png image
#
############################################################
def graph_accuracy(acc, k_val):
    x_pos = np.arange(len(k_val))
    for c in range(0, numDigits, 1):
        graph = plt.bar(x_pos, acc[c, :], align='center', color=colors[c])
        for p in graph:
            width = p.get_width()
            height = p.get_height()
            x, y = p.get_xy()
            plt.text(x + width / 2, y + height * 1.01, f'{height:.2f}' + '%', ha='center', weight='ultralight')

        plt.xticks(x_pos, k_val)
        plt.ylabel("Accuracy")
        plt.xlabel("K values")
        plt.title("Accuracy for recognizing " + str(c) + "vs. remaining possible values")
        plt.savefig(str(c) + "vs.Values.png")
        plt.show()


##############################################################
#
# Get datasets into matrix, set k values
# Train the kNN classifier and test the classifier
# Compute the error rate and the accuracy
#
##############################################################
def main():
    trainX, trainY, testX, testY = import_data_set()
    trainX_list = []
    trainY_list = []
    for i in range(trainX.shape[0]):
        image_trainX = trainX[i].reshape((28, 28))
        trainX_list.append(np.array(image_trainX).flatten().astype(int))
        trainY_list.append(int(trainY[i]))

    trainX_matrix = np.array(trainX_list)

    testX_list = []
    testY_list = []
    for i in range(testX.shape[0]):
        image_testX = testX[i].reshape((28, 28))
        testX_list.append(np.array(image_testX).flatten().astype(int))
        testY_list.append(int(testY[i]))

    testX_matrix = np.array(testX_list)

    k_values = list([1, 2, 3, 4, 5, 10, 15, 20, 30, 40, 50])

    """print("++++++++++++++++++++++++++++ Not-Binary Accuracy ++++++++++++++++++++++++++++")
    accuracy_matrix = np.zeros((10, len(k_values)))
    for k in range(0, len(k_values), 1):
        print("\n \n********************************************************************************")
        print("K = " + str(k_values[k]))
        predictionList, error_rate = kNN.knnClassifier_manager(trainX_matrix, trainY_list, testX_matrix,
                                                               k_values[k], testY_list)
        print("\nK = " + str(k_values[k]))
        print("\nError rate: " + str(error_rate) + "%")
        for j in range(0, 10, 1):
            accuracy_matrix[j, k] = kNN.accuracy_kNN_classifier(predictionList, testY_list, j)
            print("Accuracy of " + str(j) + " vs. remaining values: " + str(accuracy_matrix[j, k]) + "%\n")

    graph_accuracy(accuracy_matrix, k_values)

    print("++++++++++++++++++++++++++++ Binary Accuracy ++++++++++++++++++++++++++++")
    accuracy_matrixBinary = np.zeros((numDigits, len(k_values)))
    trainY_listBinary = np.zeros((len(trainY_list)))
    testY_listBinary = np.zeros((len(testY_list)))
    for c in range(0, numDigits, 1):
        for i in range(0, len(trainY_list), 1):
            if trainY_list[i] == c:
                trainY_listBinary[i] = int(trainY_list[i])
            else:
                trainY_listBinary[i] = int(-1)
        for j in range(0, len(testY_list), 1):
            if testY_list[j] == c:
                testY_listBinary[j] = int(testY_list[j])
            else:
                testY_listBinary[j] = int(-1)

        for k in range(0, len(k_values), 1):
            print("\n \n********************************************************************************")
            print("For K = " + str(k_values[k]))
            print("Digits = " + str(c))
            predictionListBinary, error_rate = kNN.knnClassifier_manager(trainX_matrix, trainY_listBinary, testX_matrix,
                                                                         k_values[k], testY_listBinary)
            print("\n \n********************* Recognizing " + str(c) + " against remaining class *********************")
            print("For K = " + str(k_values[k]))
            print("Digits = " + str(c))
            accuracy_matrixBinary[c, k] = kNN.accuracy_kNN_classifier(predictionListBinary, testY_listBinary, c)[0]
            print("Accuracy of " + str(c) + " vs. remaining class: " + str(accuracy_matrixBinary[c, k]) + "%\n")

    graph_accuracyBinary(accuracy_matrixBinary, k_values)"""

    print("\n\n++++++++++++++++++++++++++++ Evaluation of classifier ++++++++++++++++++++++++++++\n")
    randomTrainX, randomTrainY = get_random_subsets(trainX_matrix, trainY_list, int(percentageSubsets), int(numSubsets))
    print("\n************************ Random subsets dimensions ************************")
    print(randomTrainX.shape)
    print(randomTrainY.shape)

    k_values_array = np.array(["1", "2", "3", "4", "5", "10", "15", "20", "30", "40", "50"])
    accuracy_matrixSub = np.zeros((numSubsets, numDigits, len(k_values)))
    indexes_matrix = np.zeros((numSubsets, numDigits, len(k_values), 5))
    trainY_listBinary = np.zeros((numSubsets, len(trainY_list)))
    testY_listBinary = np.zeros((numSubsets, len(testY_list)))

    for s in range(0, numSubsets, 1):
        for c in range(0, numDigits, 1):
            for i in range(0, int(randomTrainY.shape[1]), 1):
                if randomTrainY[s, i] == c:
                    trainY_listBinary[s, i] = int(randomTrainY[s, i])
                else:
                    trainY_listBinary[s, i] = int(-1)
            for j in range(0, len(testY_list), 1):
                if testY_list[j] == c:
                    testY_listBinary[s, j] = int(testY_list[j])
                else:
                    testY_listBinary[s, j] = int(-1)

            for k in range(0, len(k_values), 1):
                print("\n \n********************************************************************************")
                print("Subsets n. " + str(s))
                print("K = " + str(k_values[k]))
                print("Digits = " + str(c))
                predictionListBinary, error_rate = kNN.knnClassifier_manager(randomTrainX[s], trainY_listBinary[s],
                                                                             testX_matrix, k_values[k],
                                                                             testY_listBinary[s])
                print("\n \n****************** Recognizing " + str(c) + " against remaining class ******************")
                print("Subsets n. " + str(s))
                print("For K = " + str(k_values[k]))
                print("Digits = " + str(c))
                accuracy_matrixSub[s, c, k], indexes_matrix[s, c, k, 0], indexes_matrix[s, c, k, 1], \
                indexes_matrix[s, c, k, 2], indexes_matrix[s, c, k, 3], indexes_matrix[s, c, k, 4] = \
                    kNN.accuracy_kNN_classifier(predictionListBinary, testY_listBinary[s], c)
                print("Accuracy of " + str(c) + " vs. remaining class: " + str(accuracy_matrixSub[s, c, k]) + "%\n")

    sumSens = np.zeros((numDigits, len(k_values)), dtype=float)
    sumSpec = np.zeros((numDigits, len(k_values)), dtype=float)
    sumPrec = np.zeros((numDigits, len(k_values)), dtype=float)
    sumRec = np.zeros((numDigits, len(k_values)), dtype=float)
    sumF1 = np.zeros((numDigits, len(k_values)), dtype=float)
    sdSens = np.zeros((numDigits, len(k_values)), dtype=float)
    sumSDspec = np.zeros((numDigits, len(k_values)), dtype=float)
    sumSDprec = np.zeros((numDigits, len(k_values)), dtype=float)
    sumSDrec = np.zeros((numDigits, len(k_values)), dtype=float)
    sumSDf1 = np.zeros((numDigits, len(k_values)), dtype=float)
    sumSDsens = np.zeros((numDigits, len(k_values)), dtype=float)
    sdSpec = np.zeros((numDigits, len(k_values)), dtype=float)
    sdPrec = np.zeros((numDigits, len(k_values)), dtype=float)
    sdRec = np.zeros((numDigits, len(k_values)), dtype=float)
    sdF1 = np.zeros((numDigits, len(k_values)), dtype=float)
    averageSens = np.zeros((numDigits, len(k_values)), dtype=float)
    averageSpec = np.zeros((numDigits, len(k_values)), dtype=float)
    averagePrec = np.zeros((numDigits, len(k_values)), dtype=float)
    averageRec = np.zeros((numDigits, len(k_values)), dtype=float)
    averageF1 = np.zeros((numDigits, len(k_values)), dtype=float)
    averageIndexesMatrix = np.empty((numDigits, len(k_values), 5), dtype='object')

    for c in range(0, numDigits, 1):
        for k in range(0, len(k_values), 1):
            for s in range(0, numSubsets, 1):
                sumSens[c, k] = sumSens[c, k] + indexes_matrix[s, c, k, 0]
                sumSpec[c, k] = sumSpec[c, k] + indexes_matrix[s, c, k, 1]
                sumPrec[c, k] = sumPrec[c, k] + indexes_matrix[s, c, k, 2]
                sumRec[c, k] = sumRec[c, k] + indexes_matrix[s, c, k, 3]
                sumF1[c, k] = sumF1[c, k] + indexes_matrix[s, c, k, 4]
            averageSens[c, k] = (sumSens[c, k] / numSubsets)
            averageSpec[c, k] = (sumSpec[c, k] / numSubsets)
            averagePrec[c, k] = (sumPrec[c, k] / numSubsets)
            averageRec[c, k] = (sumRec[c, k] / numSubsets)
            averageF1[c, k] = (sumF1[c, k] / numSubsets)
            for s in range(0, numSubsets, 1):
                sumSDsens[c, k] = sumSDsens[c, k] + pow((indexes_matrix[s, c, k, 0] - averageSens[c, k]), 2)
                sumSDspec[c, k] = sumSDspec[c, k] + pow((indexes_matrix[s, c, k, 1] - averageSpec[c, k]), 2)
                sumSDprec[c, k] = sumSDprec[c, k] + pow((indexes_matrix[s, c, k, 2] - averagePrec[c, k]), 2)
                sumSDrec[c, k] = sumSDrec[c, k] + pow((indexes_matrix[s, c, k, 3] - averageRec[c, k]), 2)
                sumSDf1[c, k] = sumSDf1[c, k] + pow((indexes_matrix[s, c, k, 3] - averageF1[c, k]), 2)

            sdSens[c, k] = math.sqrt(sumSDsens[c, k] / numSubsets)
            sdSpec[c, k] = math.sqrt(sumSDspec[c, k] / numSubsets)
            sdPrec[c, k] = math.sqrt(sumSDprec[c, k] / numSubsets)
            sdRec[c, k] = math.sqrt(sumSDrec[c, k] / numSubsets)
            sdF1[c, k] = math.sqrt(sumSDf1[c, k] / numSubsets)

            averageIndexesMatrix[c, k, 0] = str(round(averageSens[c, k], 6)) + " (" + str(round(sdSens[c, k], 6)) + ")"
            averageIndexesMatrix[c, k, 1] = str(round(averageSpec[c, k], 6)) + " (" + str(round(sdSpec[c, k], 6)) + ")"
            averageIndexesMatrix[c, k, 2] = str(round(averagePrec[c, k], 6)) + " (" + str(round(sdPrec[c, k], 6)) + ")"
            averageIndexesMatrix[c, k, 3] = str(round(averageRec[c, k], 6)) + " (" + str(round(sdRec[c, k], 6)) + ")"
            averageIndexesMatrix[c, k, 4] = str(round(averageF1[c, k], 6)) + " (" + str(round(sdF1[c, k], 6)) + ")"

    createCSV.dfToCSV(averageIndexesMatrix, k_values_array)


main()
