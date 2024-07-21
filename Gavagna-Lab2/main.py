# ----------------------------------------------------------------------------------------------------------------------
# Linear Regression
# ----------------------------------------------------------------------------------------------------------------------
#
# Veronica Gavagna
#
# ----------------------------------------------------------------------------------------------------------------------
#
# Machine Learning for Robotics 1 - 2022/2023
# Assignment 2:
# Task 1: Get data
# Task 2: Fit a linear regression model
#       2.1: One-dimensional problem without intercept on the Turkish stock exchange data
#       2.2: Compare graphically the solution obtained on different random subsets (10%) of the whole data set
#       2.3: One-dimensional problem with intercept on the Motor Trends car data, using columns mpg and weight
#       2.4: Multi-dimensional problem on the complete MTcars data, using all four columns
#            (predict mpg with the other three columns)
#       Remark 2: To make differences more visible, compare graphs of 2.2 with graph obtained without random the subsets
# Task 3: Test regression model
#       3.1: Re-run 1 from task 2 using only 5% of the data.
#            Compute the objective (mean square error) on the training data.
#            Compute the objective of the same models on the remaining 95% of the data.
#            Repeat for different training-test random splits and graph the results
#       3.2: Re-run 3 from task 2 using only 30% of the data.
#            Compute the objective (mean square error) on the training data.
#            Compute the objective of the same models on the remaining 70% of the data.
#            Repeat for different training-test random splits and graph the results.
#       3.3: Re-run 4 from task 2 using only 30% of the data.
#            Compute the objective (mean square error) on the training data.
#            Compute the objective of the same models on the remaining 70% of the data.
#            Repeat for different training-test random splits and graph the results
#
# ----------------------------------------------------------------------------------------------------------------------
#


import pandas as pd
import warnings
import numpy as np
import matplotlib.pyplot as plt
import oneDimensionalRegression as oneD
import multiDimensionalRegression as multiD

path = "/Users/veronicagavagna/Documents/PycharmProjects/Gavagna-Lab2/"
obs_turkish = "SP"
target_turkish = "MSCI_EU"
obs_cars = "weight"
target_cars = "mpg"


##############################################################
# Read csv files and create a dataframes
# RETURN:
#   Dataframes
##############################################################
def read_files_csv():
    warnings.simplefilter("ignore")
    df_mtcars = pd.read_csv(path + 'mtcarsdata-4features.csv', skiprows=1,
                            names=['Model', 'mpg', 'disp', 'hp', 'weight'])
    df_mtcars = pd.DataFrame(df_mtcars, columns=['mpg', 'disp', 'hp', 'weight'])
    read_file = pd.read_excel(path + 'data_akbilgic.xlsx')
    read_file.to_csv(path + 'data_akbilgic.csv', index=False, header=True)
    df_turkish = pd.read_csv(path + 'data_akbilgic.csv', skiprows=2,
                             names=['date', 'ISE_TL', 'ISE_USD', 'SP', 'DAX', 'FTSE', 'NIKKEI', 'BOVESPA', 'MSCI_EU',
                                    'EM'])
    df_turkish = pd.DataFrame(df_turkish, columns=['SP', 'MSCI_EU'])
    print("\n *************************** Original DataSet MT-cars *************************** \n")
    print(df_mtcars)
    print("\n *************************** Original DataSet Turkish/Istanbul *************************** \n")
    print(df_turkish)
    return df_mtcars, df_turkish


##############################################################
# Split dataset into many random dataset
# RETURN:
#   Split datasets
##############################################################
def split_dataset(percentage, ds):
    n_subsets = ds.shape[0] / (ds.shape[0] * percentage / 100)
    s = 0
    start = []
    end = []
    start.append(0)
    end.append(int(ds.shape[0] * percentage / 100))
    sub_ds = np.zeros((int(n_subsets) + 1, int(ds.shape[0] * percentage / 100), ds.shape[1]))
    while end[s] <= ds.shape[0]:
        sub_ds[s, :, :] = ds[start[s]:end[s]]
        print("\n ********************** Subset of Turkish dataset n. " + str(s + 1) + " ********************** \n")
        print(sub_ds[s])
        print(end[s])
        s += 1
        start.append(end[s - 1])
        end.append(end[s - 1] + int(ds.shape[0] * percentage / 100))
        if end[s] > ds.shape[0]:
            for m in range(start[s], ds.shape[0], 1):
                for k in range(0, ds.shape[1], 1):
                    sub_ds[s, m - start[s], k] = ds[m, k]
            print("\n ********************** Subset of Turkish dataset n. " + str(s + 1) + " ********************** \n")
            print(sub_ds[s])

    return sub_ds


##############################################################
# Compute the objective function (mean square error)
# for one-dim problem with or without intercept
# and multi-dim problem.
# Repeat 1000 times for different random subsets
# RETURN:
#   mean square error values of each iteration
#   for the two splitted dataset
##############################################################
def compute_j_mse_lists(percentage, ds, obsIndex, targetIndex, lss_mode):
    j_mse = list()
    j_mseRem = list()
    for count in range(0, 1000, 1):
        np.random.shuffle(ds)
        end_percentage = int(ds.shape[0] * percentage / 100)
        end_remaining = ds.shape[0]
        sub_dataset = ds[:end_percentage]
        subDataset_rem = ds[end_percentage:end_remaining]
        if lss_mode == 0:  # no Intercept
            w5 = oneD.compute_least_squares_sol_noIntercept(sub_dataset, obsIndex, targetIndex)
            j_mse.append(oneD.mean_square_error_function_noIntercept(sub_dataset, obsIndex, targetIndex, w5))
            j_mseRem.append(oneD.mean_square_error_function_noIntercept(subDataset_rem, obsIndex, targetIndex, w5))
        elif lss_mode == 1:  # with intercept
            w5_0, w5_1 = oneD.compute_least_squares_sol_intercept(sub_dataset, obsIndex, targetIndex)
            j_mse.append(oneD.mean_square_error_function_intercept(sub_dataset, obsIndex, targetIndex, w5_0, w5_1))
            j_mseRem.append(oneD.mean_square_error_function_intercept(subDataset_rem, obsIndex, targetIndex, w5_0,
                                                                      w5_1))
        elif lss_mode == 2:  # multidimensional
            xMat = sub_dataset[:, 1:3]
            tMat = sub_dataset[:, targetIndex]
            colToAdd = np.ones(sub_dataset.shape[0])
            xMat = np.insert(xMat, 0, colToAdd, axis=1)
            wMat = multiD.compute_least_square_matrix(xMat, tMat)
            yPred = np.dot(xMat, wMat)

            j_mse.append(multiD.mean_square_error_function(xMat, wMat, tMat, yPred))

            xMat_rem = subDataset_rem[:, 1:3]
            tMat_rem = subDataset_rem[:, targetIndex]
            colToAdd_rem = np.ones(subDataset_rem.shape[0])
            xMat_rem = np.insert(xMat_rem, 0, colToAdd_rem, axis=1)
            wMat_rem = multiD.compute_least_square_matrix(xMat_rem, tMat_rem)
            yPred_rem = np.dot(xMat_rem, wMat_rem)
            j_mseRem.append(multiD.mean_square_error_function(xMat_rem, wMat_rem, tMat_rem, yPred_rem))

        count += 1

    return j_mse, j_mseRem


##############################################################
# Main function to manage
# Task 1, 2 and 3
# of assignment 2
##############################################################
def main():
    print("\n ++++++++++++++++++++++++++ Assignment 2 ++++++++++++++++++++++++++ \n")

    print("\n ******* Task 1 ******* \n")
    dataFrame_mtcars, dataFrame_turkish = read_files_csv()
    targetIndex_turkish = dataFrame_turkish.columns.get_loc(target_turkish)
    obsIndex_turkish = dataFrame_turkish.columns.get_loc(obs_turkish)
    targetIndex_cars = dataFrame_mtcars.columns.get_loc(target_cars)
    obsIndex_cars = dataFrame_mtcars.columns.get_loc(obs_cars)
    print("\n ############## Original DataSet MT-Cars Matrix ############## \n")
    dataSet_mtcars = dataFrame_mtcars.to_numpy()
    print(dataSet_mtcars)
    print("\n ############## Original DataSet Turkish/Istanbul Matrix ############## \n")
    dataSet_turkish = dataFrame_turkish.to_numpy()
    print(dataSet_turkish)

    print("\n ******* Task 2.1 ******* \n")
    print("\n ############## Compute least squares solution to linear regression on dataset Turkish ############## \n")
    wTurkish = oneD.compute_least_squares_sol_noIntercept(dataSet_turkish, targetIndex_turkish, obsIndex_turkish)
    print("wTurkish = " + str(wTurkish))

    print("\n ############## Graph one-dimensional linear regression on dataset Turkish ############## \n")
    plt.scatter(dataSet_turkish[:, obsIndex_turkish], dataSet_turkish[:, targetIndex_turkish], label="dataset Turkish",
                color="green", marker="*")
    x = np.linspace(-0.07, 0.07, 100)
    y = wTurkish * x
    plt.plot(x, y, '-r', label='t=w*x with w = ' + str(wTurkish))
    plt.xlabel(obs_turkish)
    plt.ylabel(target_turkish)
    plt.title('The least squares solution - Turkish dataset')
    plt.grid()
    plt.legend(loc='upper left')
    plt.show()

    print("\n ******* Task 2.2 ******* \n")
    print("\n ############## Random subsets 10% of Turkish dataset matrix ############## \n")
    dataSet_turkish_random = dataSet_turkish.copy()
    np.random.shuffle(dataSet_turkish_random)
    subDataset_turkish_random = split_dataset(10, dataSet_turkish_random)

    print("\n ############## Graph one-dimensional linear regression on random subsets Turkish ############## \n")
    wSubTurkish_random = list()
    for i in range(0, subDataset_turkish_random.shape[0] - 1, 1):
        wSubTurkish_random.append(oneD.compute_least_squares_sol_noIntercept(subDataset_turkish_random[i, :, :],
                                                                             obsIndex_turkish, targetIndex_turkish))

    for j in range(0, subDataset_turkish_random.shape[0] - 2, 2):
        plt.scatter(subDataset_turkish_random[j, :, obsIndex_turkish],
                    subDataset_turkish_random[j, :, targetIndex_turkish], label="subset {0}".format(str(j)),
                    color="green", marker="*")
        x1 = np.linspace(-0.07, 0.07, 100)
        y1 = wSubTurkish_random[j] * x1
        plt.plot(x1, y1, '-b', label=('w{0}= {1}'.format(str(j), str(wSubTurkish_random[j]))))
        x2 = np.linspace(-0.07, 0.07, 100)
        y2 = wSubTurkish_random[j + 1] * x2
        plt.plot(x2, y2, "-c", label=('w{0}= {1}'.format(str(j + 1), str(wSubTurkish_random[j + 1]))))
        plt.xlabel(obs_turkish)
        plt.ylabel(target_turkish)
        plt.title('The least squares solution - Turkish random subsets {0} vs. {1}'.format(str(j), str(j + 1)))
        plt.grid()
        plt.legend(loc='upper left')
        plt.show()

    print("\n ******* Task 2 remark 2 ******* \n")
    print("\n ############## Not-random subsets 10% of Turkish/Istanbul dataset matrix ############## \n")
    subDataset_turkish = split_dataset(10, dataSet_turkish)

    print("\n ############## Graph one-dimensional linear regression on not-random subsets Turkish ############## \n")
    wSubTurkish = list()
    for i in range(0, subDataset_turkish.shape[0] - 1, 1):
        wSubTurkish.append(oneD.compute_least_squares_sol_noIntercept(subDataset_turkish[i, :, :], obsIndex_turkish,
                                                                      targetIndex_turkish))

    for j in range(0, subDataset_turkish.shape[0] - 2, 2):
        plt.scatter(subDataset_turkish[j, :, obsIndex_turkish], subDataset_turkish[j, :, targetIndex_turkish],
                    label="subset {0}".format(str(j)), color="blue", marker="*")
        x1 = np.linspace(-0.07, 0.07, 100)
        y1 = wSubTurkish[j] * x1
        plt.plot(x1, y1, '-g', label=('w{0}= {1}'.format(str(j), str(wSubTurkish[j]))))
        x2 = np.linspace(-0.07, 0.07, 100)
        y2 = wSubTurkish[j + 1] * x2
        plt.plot(x2, y2, "-y", label=('w{0}= {1}'.format(str(j + 1), str(wSubTurkish[j + 1]))))
        plt.xlabel(obs_turkish)
        plt.ylabel(target_turkish)
        plt.title('The least squares solution - Turkish not-random subsets {0} vs. {1}'.format(str(j), str(j + 1)))
        plt.grid()
        plt.legend(loc='upper left')
        plt.show()

    print("\n ******* Task 2.3 ******* \n")
    print("\n ############ Compute least squares solution to linear regression on dataset Motor Trends ############ \n")
    wMotor0, wMotor1 = oneD.compute_least_squares_sol_intercept(dataSet_mtcars, obsIndex_cars, targetIndex_cars)
    print('w0 = ' + str(wMotor0))
    print('w1 = ' + str(wMotor1))

    print("\n ############## Graph one-dimensional linear regression on dataset Motor Trends ############## \n")
    plt.scatter(dataSet_mtcars[:, obsIndex_cars], dataSet_mtcars[:, targetIndex_cars], label="dataset Motor Trends",
                color="cyan", marker="*")
    x = np.linspace(1, 6, 100)
    y = wMotor1 * x + wMotor0
    plt.plot(x, y, '-r', label=('t=w0 + w1*x\n' + 'w0= ' + str(wMotor0) + '\nw1= ' + str(wMotor1)))
    plt.xlabel(obs_cars)
    plt.ylabel(target_cars)
    plt.title('The least squares solution - Motor Trends dataset')
    plt.grid()
    plt.legend(loc='lower left')
    plt.show()

    print("\n ******* Task 2.4 ******* \n")
    print("\n ############## Compute least squares solution to multi-dimensional linear regression on dataset Motor "
          "Trends ############## \n")
    xMatrix = dataFrame_mtcars[['disp', 'hp', 'weight']].copy()
    xMatrix = xMatrix.to_numpy()
    tMatrix = dataFrame_mtcars[['mpg']].copy()
    tMatrix = tMatrix.to_numpy()
    columnToAdd = np.ones(dataSet_mtcars.shape[0])
    xMatrix = np.insert(xMatrix, 0, columnToAdd, axis=1)
    wMatrix = multiD.compute_least_square_matrix(xMatrix, tMatrix)
    print("\n ######### Mpg predicted through multi-dimensional linear regression on dataset Motor Trends ######### \n")
    yPredicted = np.dot(xMatrix, wMatrix)
    print(yPredicted)

    print("\n ******* Task 3.1 ******* \n")
    j_mseTurkish, j_mseRemainingTurkish = compute_j_mse_lists(5, dataSet_turkish_random,
                                                              obsIndex_turkish, targetIndex_turkish, 0)

    print("\n ######## Graph mean square error for 1000 different subsets of the 5% of the dataset Turkish ######## \n")
    print("j MSE Turkish of 5%")
    print(j_mseTurkish)

    j_mseT5_round = [round(elem, 7) for elem in j_mseTurkish]
    figT5, axT5 = plt.subplots(1, 1)
    nT5, bins_listT5, patchesT5 = plt.hist(j_mseT5_round, bins=10, color='orange', rwidth=0.5)
    plt.xticks(rotation=45)
    bins_listT5 = np.delete(bins_listT5, (len(bins_listT5) - 1))
    bins_list_roundT5 = [round(elem, 7) for elem in bins_listT5]
    scientific_listT5 = list()
    for x in bins_list_roundT5:
        scientific_listT5.append(f'{x:.3e}'.format(bins_list_roundT5))

    bins_stringT5 = [str(s) for s in scientific_listT5]
    plt.grid()
    axT5.set_xticks(bins_listT5)
    axT5.set_xticklabels(bins_stringT5)
    plt.xlabel("Objective function value")
    plt.ylabel("Number of occurrences")
    plt.title("Histogram of MSE values for 1000 different data subsets 5% \n Turkish - One-dimensional")
    plt.show()

    print(
        "\n ######## Graph mean square error for 1000 different subsets of the 95% of the dataset Turkish ######## \n")
    print("\n j MSE Turkish of the remaining")
    print(j_mseRemainingTurkish)

    j_mseRemT_round = [round(elem, 7) for elem in j_mseRemainingTurkish]
    figT95, axT95 = plt.subplots(1, 1)
    nT95, bins_listT95, patchesT95 = plt.hist(j_mseRemT_round, bins=20, color='green', rwidth=0.5, range=[5e-05, 2e-04])
    plt.xticks(rotation=45)
    bins_listT95 = np.delete(bins_listT95, (len(bins_listT95) - 1))
    bins_list_roundT95 = [round(elem, 7) for elem in bins_listT95]
    scientific_listT95 = list()
    for x in bins_list_roundT95:
        scientific_listT95.append(f'{x:.3e}'.format(bins_list_roundT95))

    bins_stringT95 = [str(s) for s in scientific_listT95]
    plt.grid()
    axT95.set_xticks(bins_listT95)
    axT95.set_xticklabels(bins_stringT95)
    plt.xlabel("Objective function value")
    plt.ylabel("Number of occurrences")
    plt.title("Histogram of MSE values for 1000 different data subsets 95% \n Turkish - One-dimensional")
    plt.show()

    print("\n ******* Task 3.2 ******* \n")
    dataSet_mtcars_random = dataSet_mtcars.copy()
    np.random.shuffle(dataSet_mtcars_random)
    j_mseCars, j_mseRemainingCars = compute_j_mse_lists(30, dataSet_mtcars_random, obsIndex_cars,
                                                        targetIndex_cars, 1)

    print("\n ######## Graph mean square error for 1000 different subsets of the 30% of the dataset MTcars ######## \n")
    print("j MSE MTcars of 30%")
    print(j_mseCars)

    j_mseC5_round = [round(elem, 3) for elem in j_mseCars]
    figC5, axC5 = plt.subplots(1, 1)
    nC5, bins_listC5, patchesC5 = plt.hist(j_mseC5_round, bins=10, color='red', rwidth=0.5)
    plt.xticks(rotation=45)
    bins_listC5 = np.delete(bins_listC5, (len(bins_listC5) - 1))
    bins_list_roundC5 = [round(elem, 3) for elem in bins_listC5]
    plt.grid()
    axC5.set_xticks(bins_listC5)
    axC5.set_xticklabels(bins_list_roundC5)
    plt.xlabel("Objective function value")
    plt.ylabel("Number of occurrences")
    plt.title("Histogram of MSE values for 1000 different data subsets 30% \n MTcars - One-dimensional")
    plt.show()

    print(
        "\n ######### Graph mean square error for 1000 different subsets of the 70% of the dataset MTcars ######### \n")
    print("\n j MSE Cars of the remaining")
    print(j_mseRemainingCars)

    j_mseRemC_round = [round(elem, 3) for elem in j_mseRemainingCars]
    figC95, axC95 = plt.subplots(1, 1)
    nC95, bins_listC95, patchesC95 = plt.hist(j_mseRemC_round, bins=20, color='blue', rwidth=0.5, range=[6, 60])
    plt.xticks(rotation=45)
    bins_listC95 = np.delete(bins_listC95, (len(bins_listC95) - 1))
    bins_list_roundC95 = [round(elem, 3) for elem in bins_listC95]
    plt.grid()
    axC95.set_xticks(bins_listC95)
    axC95.set_xticklabels(bins_list_roundC95)
    plt.xlabel("Objective function value")
    plt.ylabel("Number of occurrences")
    plt.title("Histogram of MSE values for 1000 different data subsets 70% \n MTcars - One-dimensional")
    plt.show()

    print("\n ******* Task 3.3 ******* \n")
    dataSet_mtcars_random = dataSet_mtcars.copy()
    np.random.shuffle(dataSet_mtcars_random)
    j_mseCars_MD, j_mseRemainingCars_MD = compute_j_mse_lists(30, dataSet_mtcars_random, obsIndex_cars,
                                                              targetIndex_cars, 2)

    print("\n ########## Graph mean square error for 1000 different subsets of the 30% of the dataset MTcars - "
          "Multi dimensional########## \n")
    print("j MSE MTcars of 30%")
    print(j_mseCars)

    j_mseC5_round_MD = [round(elem, 3) for elem in j_mseCars_MD]
    figC5_MD, axC5_MD = plt.subplots(1, 1)
    nC5_MD, bins_listC5_MD, patchesC5 = plt.hist(j_mseC5_round_MD, bins=10, color='magenta', rwidth=0.5)
    plt.xticks(rotation=45)
    bins_listC5_MD = np.delete(bins_listC5_MD, (len(bins_listC5_MD) - 1))
    bins_list_roundC5_MD = [round(elem, 3) for elem in bins_listC5_MD]
    plt.grid()
    axC5_MD.set_xticks(bins_listC5_MD)
    axC5_MD.set_xticklabels(bins_list_roundC5_MD)
    plt.xlabel("Objective function value")
    plt.ylabel("Number of occurrences")
    plt.title("Histogram of MSE values for 1000 different data subsets 30% \n MTcars - Multi dimensional")
    plt.show()

    print(
        "\n ########## Graph mean square error for 1000 different subsets of the 70% of the dataset MTcars - "
        "Multi dimensional ########## \n")
    print("\n j MSE Cars of the remaining")
    print(j_mseRemainingCars_MD)

    j_mseRemC_round_MD = [round(elem, 3) for elem in j_mseRemainingCars_MD]
    figC95_MD, axC95_MD = plt.subplots(1, 1)
    nC95_MD, bins_listC95_MD, patchesC95_MD = plt.hist(j_mseRemC_round_MD, bins=20, color='cyan', rwidth=0.5)
    plt.xticks(rotation=45)
    bins_listC95_MD = np.delete(bins_listC95_MD, (len(bins_listC95_MD) - 1))
    bins_list_roundC95_MD = [round(elem, 3) for elem in bins_listC95_MD]
    int_listC95_MD = list()
    for x in bins_list_roundC95_MD:
        int_listC95_MD.append(f'{x:.2f}'.format(bins_list_roundC95_MD))

    bins_stringC95_MD = [str(s) for s in int_listC95_MD]
    plt.grid()
    axC95_MD.set_xticks(bins_listC95_MD)
    axC95_MD.set_xticklabels(bins_stringC95_MD)
    plt.xlabel("Objective function value")
    plt.ylabel("Number of occurrences")
    plt.title("Histogram of MSE values for 1000 different data subsets 70% \n MTcars - Multi dimensional")
    plt.show()


main()
