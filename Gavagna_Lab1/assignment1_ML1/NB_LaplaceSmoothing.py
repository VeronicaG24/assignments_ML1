# ----------------------------------------------------------------------------------------------------------------------
# Naive Bayes Classifier - Laplace Smoothing
# ----------------------------------------------------------------------------------------------------------------------
# Veronica Gavagna
# ----------------------------------------------------------------------------------------------------------------------
#

import numpy as np


##############################################################
# Check if each value of the test set is valid,
# otherwise it deletes the pattern corresponding to that value
# RETURN:
#   test set with all values valid
##############################################################
def check_testSet_valuesL(no_copy, test_set, num_levels):
    test_set = np.delete(test_set, 0, 0)
    for i in range(test_set.shape[0]):
        for j in range(test_set.shape[1]):
            for n in range(len(no_copy[j])):
                if test_set[i, j] not in no_copy[j]:
                    print("*** value not in training set --> delete row ***")
                    testSet_new = np.delete(test_set, j, 0)
                    test_set = testSet_new.copy()

    test_set = np.insert(test_set, 0, num_levels, axis=0)
    return test_set


##############################################################
# Search for all the possible values (levels) for
# each level,
# call check_testSet_values
# RETURN:
#   a matrix with only the possible values unique
#   and zeros where there is no values,
#   target values unique,
#   max number of values for the all categories
##############################################################
def get_unique_valuesL(test_s, trainingSet, targetIndex, numLevels):
    target_list = list()
    no_copy = list()
    dim_max_values = 0
    train_set_noRaw1 = np.delete(trainingSet, 0, 0)
    for j in range(train_set_noRaw1.shape[1]):
        elements = list()
        for n in train_set_noRaw1.T[j]:
            if n not in elements:
                elements.append(n)
        if dim_max_values <= len(elements):
            dim_max_values = len(elements)
        no_copy.append(elements)

    test_set_ok = check_testSet_valuesL(no_copy, test_s, numLevels)

    target_list.append(no_copy[targetIndex])
    target = np.array(target_list, dtype=int)

    for s in range(len(no_copy)):
        while len(no_copy[s]) != dim_max_values:
            no_copy[s].append(0)

    header = np.array(no_copy, dtype=int)

    print("---------------------------------------------")
    print(header)
    print("---------------------------------------------")
    return header, target, dim_max_values, test_set_ok


##############################################################
# Calculate the likelihood probability for each pattern of the
# training set with the Laplace Smoothing formula
# RETURN:
#   3D matrix with all the likelihood probabilities
##############################################################
def calculate_lh_probabilitiesL(training_set, targetIndex, a, testSetL, numLevels):
    rows_training_set, columns_training_set = training_set.shape

    header, target, dim_max_values, test_set_ok = get_unique_valuesL(testSetL, training_set, targetIndex, numLevels)

    prob_target = np.zeros(target.shape[1], dtype=float)
    total_target = np.zeros(target.shape[1], dtype=int)
    for k in range(0, target.shape[1], 1):
        for i in range(1, training_set.shape[0], 1):
            if target[0, k] == training_set[i, targetIndex]:
                total_target[k] = total_target[k] + 1
        prob_target[k] = (total_target[k] + a) / (training_set.shape[0] + (a * numLevels[targetIndex]))

    print("---------------------------------------------")
    print(prob_target)
    print("---------------------------------------------")

    likelihood_prob = np.zeros((target.shape[1], columns_training_set, dim_max_values), dtype=float)
    total_value = np.zeros((target.shape[1], dim_max_values), dtype=int)
    for k in range(0, target.shape[1], 1):
        for j in range(0, columns_training_set, 1):
            for h in range(0, dim_max_values, 1):
                for i in range(1, rows_training_set, 1):
                    if training_set[i, j] == header[j, h] and training_set[i, targetIndex] == header[targetIndex, k]:
                        total_value[k, h] = total_value[k, h] + 1
                likelihood_prob[k, j, h] = (total_value[k, h] + a) / (total_target[k] + (a * numLevels[j]))
            total_value = np.zeros((target.shape[1], dim_max_values), dtype=int)

    print("---------------------------------------------")
    print(likelihood_prob)
    print("---------------------------------------------")
    return likelihood_prob, header, target, test_set_ok, prob_target


##############################################################
# Test the classifier using the test set
##############################################################
def test_NB_classifierL(testSetL, targetValuesL, possibleValuesL, likelihoodProbL, targetIndex):
    posterior_prob = np.ones(((testSetL.shape[0] - 1), targetValuesL.shape[1]), dtype=float)
    for k in range(0, targetValuesL.shape[1], 1):
        for i in range(1, testSetL.shape[0], 1):
            for j in range(0, testSetL.shape[1] - 1, 1):
                for h in range(0, possibleValuesL.shape[1], 1):
                    if testSetL[i, j] == possibleValuesL[j, h]:
                        posterior_prob[i - 1, k] = posterior_prob[i - 1, k] * likelihoodProbL[k, j, h]

    print("---------------------------------------------")
    print(posterior_prob)
    print("---------------------------------------------")

    for i in range(1, testSetL.shape[0], 1):
        max_prob_index = np.argmax(posterior_prob[i-1, :], axis=0)
        testSetL[i, targetIndex] = targetValuesL[0, max_prob_index]


##############################################################
# Calculate the error rate by comparing with the old test set
# RETURN:
#   the error rate
##############################################################
def compute_error_rateL(testSet_oldL, targetIndex, testSetL):
    error_rate = 0.
    if testSet_oldL[2, targetIndex] != 0:
        print("*** check test: new test set ***")
        print(testSetL)
        print("*** check test set: old test set ***")
        print(testSet_oldL)
        total_error = 0
        for i in range(1, testSet_oldL.shape[0], 1):
            if testSetL[i, targetIndex] != testSet_oldL[i, targetIndex]:
                total_error = total_error + 1

        error_rate = total_error / (testSet_oldL.shape[0] - 1)
        print("---------------------------------------------")
        print("Error rate: ")
        print(error_rate)
    else:
        print("*** not possible to check classifier results ***")

    return error_rate
