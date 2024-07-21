# ----------------------------------------------------------------------------------------------------------------------
# kNN Classifier
# ----------------------------------------------------------------------------------------------------------------------
# Veronica Gavagna
# ----------------------------------------------------------------------------------------------------------------------
#

import numpy as np
import matplotlib.pyplot as plt
from inspect import signature
import pandas as pd

num_mandatory_arguments = int(4)


##############################################################
# Compute the euclidean distance between two points
# RETURN:
#   euclidean distance
##############################################################
def euclidean_distance(p1, p2):
    dist_euclidean = np.sqrt(np.sum((p1 - p2) ** 2))

    return dist_euclidean


##############################################################
# List the neighbors given the distances and the number
# RETURN:
#   array of the closest neighbors
##############################################################
def get_neighbors(distances, num_neighbors):
    sorted_distances = sorted(distances, key=lambda tup: tup[0])
    neighbors = np.array(sorted_distances[:num_neighbors])

    return neighbors


##############################################################
# List the neighbors given the distances and the number
# RETURN:
#   array of the closest neighbors
##############################################################
def get_prediction(neighbors):
    freq = np.unique(neighbors[:, 1], return_counts=True)
    labels, counts = freq
    prediction = labels[counts.argmax()]

    return prediction


##############################################################
# Compute the prediction of the label
# RETURN:
#   predicted values
##############################################################
def knn_classifier(x_train, y_train, test_point, k):
    distEuclidean = []

    for i in range(x_train.shape[0]):
        data_point = x_train[i]
        label = y_train[i]
        distEuclidean.append((euclidean_distance(test_point, data_point), label))

    k_nearest_neighbors = get_neighbors(distEuclidean, k)
    predicted_value = int(get_prediction(k_nearest_neighbors))

    return predicted_value


##################################################################
# Check the number of arguments passed to the classifier function
# RETURN:
#   true if the comparison is possible, false if not
#   raise exception if the number is less than the mandatory
##################################################################
def check_num_arguments(n_par):
    if n_par >= num_mandatory_arguments:
        print("Number of arguments correct: " + str(n_par))
        if n_par > num_mandatory_arguments:
            comp = True
            print("Comparison is possible")
            return comp
        elif n_par == num_mandatory_arguments:
            comp = False
            print("Comparison not possible")
    else:
        raise Exception("Sorry, number of arguments not valid")

    return comp


##############################################################
# Check the dimensions of two arrays are equal
# RETURN:
#   true if dimensions are equal, false if not
##############################################################
def check_dimensions(check1, check2):
    if check1 == check2:
        check = True
    else:
        raise Exception("Sorry, dimensions not equal")

    return check


##############################################################
# Check if the number of neighbors (k) is valid
# RETURN:
#   true if it is valid, false if not
##############################################################
def check_num_neighbors(num, train):
    if 0 < num <= train.shape[0]:
        check = True
    else:
        raise Exception("Sorry, number of neighbors not valid")

    return check


#################################
# Compute the error rate
# RETURN:
#   value of the error rate
#################################
def compute_error_rate(expected, predicted):
    count = 0
    for i in range(0, len(predicted), 1):
        if not(expected[i] == predicted[i]):
            count += 1

    error = count / len(predicted)

    return error


####################################################################
# Manage the kNN classifier, train, test and compute the error rate
# RETURN:
#   labels of the test list and the error rate
####################################################################
def knnClassifier_manager(train_x, train_y, test_x, k, *args):
    arguments = signature(knn_classifier)
    params = arguments.parameters
    num_arguments = len(params)+len(args)

    comparison = check_num_arguments(num_arguments)

    columns_ok = check_dimensions(train_x.shape[1], test_x.shape[1])
    if columns_ok:
        print("OK: number of columns of the second matrix equals the number of columns of the first matrix")

    num_neighbors_ok = check_num_neighbors(k, train_x)
    if num_neighbors_ok:
        print("OK: number of neighbors valid\n")

    if comparison:
        test_y = args[0]
        print("Print 10 image for example")
        """for i in range(0, 10, 1):
            test = test_x[i]
            im = test.reshape((28, 28))
            plt.figure()
            plt.imshow(im, cmap='gray')
            label_test = knn_classifier(train_x, train_y, test, k)
            plt.title("Expected: " + str(int(test_y[i])) + " - Predicted: " + str(label_test))
            plt.show()"""
        label_test_list = list()
        for i in range(0, test_x.shape[0], 1):
            test = test_x[i]
            label_test_list.append(knn_classifier(train_x, train_y, test, k))
            # print(str(i+1) + ") Expected: " + str(int(test_y[i])) + " - Predicted: " + str(label_test_list[i]))
        error_rate = compute_error_rate(test_y, label_test_list)
        error_rate = round(error_rate*100, 2)
        return label_test_list, error_rate
    else:
        for i in range(0, 10, 1):
            test = test_x[i]
            im = test.reshape((28, 28))
            plt.figure()
            plt.imshow(im, cmap='gray')
            label_test = knn_classifier(train_x, train_y, test, k)
            plt.title("No comparison\nPredicted: " + str(label_test))
            plt.show()

        label_test_list = list()
        for i in range(0, test_x.shaoe[0], 1):
            test = test_x[i]
            label_test_list = knn_classifier(train_x, train_y, test, k)
            print("No comparison - Predicted: " + str(label_test_list[i]))
        return label_test_list, 0


##############################################################
# Test the kNN classifier with different values of k
# Compute the accuracy
# RETURN:
#   accuracy
##############################################################
def accuracy_kNN_classifier(pred_list, test_y, num):
    total_num = 0
    true_neg = 0
    true_pos = 0
    false_pos = 0
    false_neg = 0
    confusion_matrix = np.empty((2, 2), dtype='object')
    for j in range(0, len(pred_list), 1):
        if test_y[j] == num:
            total_num += 1
        if test_y[j] == -1 and pred_list[j] == test_y[j]:
            true_neg += 1
        if test_y[j] == num and pred_list[j] == test_y[j]:
            true_pos += 1
        if test_y[j] == -1 and pred_list[j] == num:
            false_pos += 1
        if test_y[j] == num and pred_list[j] == -1:
            false_neg += 1

    confusion_matrix[0, 0] = str(true_neg)
    confusion_matrix[0, 1] = str(false_pos)
    confusion_matrix[1, 0] = str(false_neg)
    confusion_matrix[1, 1] = str(true_pos)

    df_confMatrix = pd.DataFrame(confusion_matrix, columns=['Negative', 'Positive'])
    df_confMatrix.to_csv('confusionMatrix' + str(num) + '.csv', index=False, sep=',')

    accuracy = true_pos / total_num
    accuracy = round(accuracy * 100, 2)

    sensitivity = round(true_pos / (true_pos + true_neg), 3)
    specificity = round(true_neg / (true_neg + false_pos), 3)
    precision = round(true_pos / (true_pos + false_pos), 3)
    recall = round(true_pos / (true_pos + false_neg), 3)
    f1 = round(2 * (precision * recall / (precision + recall)), 3)

    return accuracy, sensitivity, specificity, precision, recall, f1
