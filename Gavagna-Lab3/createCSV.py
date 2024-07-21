# ----------------------------------------------------------------------------------------------------------------------
# create the .csv
# ----------------------------------------------------------------------------------------------------------------------
# Veronica Gavagna
# ----------------------------------------------------------------------------------------------------------------------
#

import pandas as pd


############################################################
#
# Crate the data frames from a matrices
# Save the .csv file
#
############################################################
def dfToCSV(averageIndexesMatrix, k_values_array):
    df_0 = pd.DataFrame(averageIndexesMatrix[0],
                        columns=['Average_Sensitivity', 'Average_Specificity', 'Average_Precision',
                                 'Average_Recall', 'Average_F1'])
    df_0.insert(loc=0, column='k', value=k_values_array)
    df_0.to_csv('table0_average.csv', index=False, sep=',')

    df_1 = pd.DataFrame(averageIndexesMatrix[1],
                        columns=['Average_Sensitivity', 'Average_Specificity', 'Average_Precision',
                                 'Average_Recall', 'Average_F1'])
    df_1.insert(loc=0, column='k', value=k_values_array)
    df_1.to_csv('table1_average.csv', index=False, sep=',')

    df_2 = pd.DataFrame(averageIndexesMatrix[2],
                        columns=['Average_Sensitivity', 'Average_Specificity', 'Average_Precision',
                                 'Average_Recall', 'Average_F1'])
    df_2.insert(loc=0, column='k', value=k_values_array)
    df_2.to_csv('table2_average.csv', index=False, sep=',')

    df_3 = pd.DataFrame(averageIndexesMatrix[3],
                        columns=['Average_Sensitivity', 'Average_Specificity', 'Average_Precision',
                                 'Average_Recall', 'Average_F1'])
    df_3.insert(loc=0, column='k', value=k_values_array)
    df_3.to_csv('table3_average.csv', index=False, sep=',')

    df_4 = pd.DataFrame(averageIndexesMatrix[4],
                        columns=['Average_Sensitivity', 'Average_Specificity', 'Average_Precision',
                                 'Average_Recall', 'Average_F1'])
    df_4.insert(loc=0, column='k', value=k_values_array)
    df_4.to_csv('table4_average.csv', index=False, sep=',')

    df_5 = pd.DataFrame(averageIndexesMatrix[5],
                        columns=['Average_Sensitivity', 'Average_Specificity', 'Average_Precision',
                                 'Average_Recall', 'Average_F1'])
    df_5.insert(loc=0, column='k', value=k_values_array)
    df_5.to_csv('table5_average.csv', index=False, sep=',')

    df_6 = pd.DataFrame(averageIndexesMatrix[6],
                        columns=['Average_Sensitivity', 'Average_Specificity', 'Average_Precision',
                                 'Average_Recall', 'Average_F1'])
    df_6.insert(loc=0, column='k', value=k_values_array)
    df_6.to_csv('table6_average.csv', index=False, sep=',')

    df_7 = pd.DataFrame(averageIndexesMatrix[7],
                        columns=['Average_Sensitivity', 'Average_Specificity', 'Average_Precision',
                                 'Average_Recall', 'Average_F1'])
    df_7.insert(loc=0, column='k', value=k_values_array)
    df_7.to_csv('table7_average.csv', index=False, sep=',')

    df_8 = pd.DataFrame(averageIndexesMatrix[8],
                        columns=['Average_Sensitivity', 'Average_Specificity', 'Average_Precision',
                                 'Average_Recall', 'Average_F1'])
    df_8.insert(loc=0, column='k', value=k_values_array)
    df_8.to_csv('table8_average.csv', index=False, sep=',')

    df_9 = pd.DataFrame(averageIndexesMatrix[9],
                        columns=['Average_Sensitivity', 'Average_Specificity', 'Average_Precision',
                                 'Average_Recall', 'Average_F1'])
    df_9.insert(loc=0, column='k', value=k_values_array)
    df_9.to_csv('table9_average.csv', index=False, sep=',')
