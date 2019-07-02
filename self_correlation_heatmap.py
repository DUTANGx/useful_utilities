import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from matplotlib import pyplot as plt
import seaborn as sns


def correlation_measures(data_generator, steps, feature_num, file_dir):
    """
    calculate the average correlations between features for all data in data generator
    :param data_generator: generator, yield feature_num * time_steps matrix at each step
    :param steps: int
    :param feature_num: int
    :param file_dir: string
    """
    data_matrix = np.zeros((feature_num, feature_num))
    while steps:
        data = next(data_generator)
        data_value = np.zeros((feature_num, feature_num))
        for i in range(feature_num):
            for j in range(i + 1, feature_num):
                # only consider lower triangular matrix
                data_value[i, j] = pearsonr(data[i], data[j])[0]
        data_matrix += data_value

    # calculate average
    data_matrix /= steps
    new_df = pd.DataFrame(data_matrix, columns=range(feature_num), index=range(feature_num))
    new_df.to_csv(file_dir)


def heat_map(dataframe, file_name, figsize=(50, 25), if_tri=True):
    """
    plot heat map using the dataframe generated from the function above
    :param dataframe: pandas dataframe
    :param file_name: string
    :param figsize: tuple
    :param if_tri: Bool, whether to plot lower triangular matrix
    """
    plt.figure(figsize=figsize)
    if if_tri:
        mask = np.zeros_like(dataframe.values)
        mask[np.triu_indices_from(mask)] = True
        sns.heatmap(dataframe, mask=mask, robust=True, annot=True, linewidths=1)
    else:
        sns.heatmap(dataframe, robust=True, annot=True, linewidths=1)
    plt.savefig(file_name)
