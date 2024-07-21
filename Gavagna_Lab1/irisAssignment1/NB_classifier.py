# ----------------------------------------------------------------------------------------------------------------------
# Naive Bayes Classifier
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
def check_testSet_values(no_copy, test_set):
    for i in range(test_set.shape[0]):
        for j in range(test_set.shape[1]):
            for n in range(len(no_copy[j])):
                if test_set[i, j] not in no_copy[j]:
                    print("*** value not in training set --> delete row ***")
                    testSet_new = np.delete(test_set, j, 0)
                    test_set = testSet_new.copy()

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
def get_unique_values(test_s, trainingSet, targetIndex):
    target_list = list()
    no_copy = list()
    dim_max_values = 0
    for j in range(0, trainingSet.shape[1], 1):
        elements = list()
        for n in trainingSet.T[j]:
            if n not in elements:
                elements.append(n)
        if dim_max_values <= len(elements):
            dim_max_values = len(elements)
        no_copy.append(elements)

    test_set_ok = check_testSet_values(no_copy, test_s)

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
# training set
# RETURN:
#   3D matrix with all the likelihood probabilities
##############################################################
def calculate_lh_probabilities(training_set, testSet, targetIndex):

    header, target, dim_max_values, test_set_ok = get_unique_values(testSet, training_set, targetIndex)

    prob_target = np.zeros(target.shape[1], dtype=float)
    total_target = np.zeros(target.shape[1], dtype=int)
    for k in range(0, target.shape[1], 1):
        for i in range(training_set.shape[0]):
            if target[0, k] == training_set[i, targetIndex]:
                total_target[k] = total_target[k] + 1
        prob_target[k] = total_target[k] / training_set.shape[0]

    print("---------------------------------------------")
    print(prob_target)
    print("---------------------------------------------")

    likelihood_prob = np.zeros((target.shape[1], training_set.shape[1]-1, dim_max_values), dtype=float)
    total_value = np.zeros((target.shape[1], dim_max_values), dtype=int)
    for k in range(0, target.shape[1], 1):
        for j in range(0, training_set.shape[1]-1, 1):
            for h in range(0, dim_max_values, 1):
                for i in range(0, training_set.shape[0], 1):
                    if header[j, h] == 0:
                        likelihood_prob[k, j, h] = 0.
                    if training_set[i, j] == header[j, h] and training_set[i, targetIndex] == header[targetIndex, k] \
                            and header[j, h] != 0:
                        total_value[k, h] = total_value[k, h] + 1
                likelihood_prob[k, j, h] = total_value[k, h] / total_target[k]
            total_value = np.zeros((target.shape[1], dim_max_values), dtype=int)

    print("---------------------------------------------")
    print(likelihood_prob)
    print("---------------------------------------------")
    return likelihood_prob, header, target, test_set_ok, prob_target


##############################################################
# Test the classifier using the test set
##############################################################
def test_NB_classifier(testSet, targetValues, likelihoodProb, possibleValues, targetIndex):
    posterior_prob = np.ones((testSet.shape[0], targetValues.shape[1]), dtype=float)
    for k in range(0, targetValues.shape[1], 1):
        for i in range(0, testSet.shape[0], 1):
            for j in range(0, testSet.shape[1] - 1, 1):
                for h in range(0, possibleValues.shape[1], 1):
                    if testSet[i, j] == possibleValues[j, h]:
                        posterior_prob[i, k] = posterior_prob[i, k] * likelihoodProb[k, j, h]

    print("---------------------------------------------")
    print(posterior_prob)
    print("---------------------------------------------")

    for i in range(0, testSet.shape[0], 1):
        max_prob_index = np.argmax(posterior_prob[i, :], axis=0)
        testSet[i, targetIndex] = targetValues[0, max_prob_index]


##############################################################
# Calculate the error rate by comparing with the old test set
# RETURN:
#   the error rate
##############################################################
def compute_error_rate(testSet_old, targetIndex, testSet):
    error_rate = 0.
    if testSet_old[0, targetIndex] != 0:
        print("*** check test: new test set ***")
        print(testSet)
        print("*** check test set: old test set ***")
        print(testSet_old)
        total_error = 0
        for i in range(testSet_old.shape[0]):
            if testSet[i, targetIndex] != testSet_old[i, targetIndex]:
                total_error = total_error + 1

        error_rate = total_error / testSet_old.shape[0]
        print("Error rate: ")
        print(error_rate)
    else:
        print("*** not possible to check classifier results ***")

    return error_rate
