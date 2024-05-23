import math
import os

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib import ticker
from scipy.stats import t
import json

import tikzplotlib

plt.style.use('fivethirtyeight')
sns.set_style(rc={"figure.facecolor":"white"})

#styles = ['Solarize_Light2', '_classic_test_patch', '_mpl-gallery', '_mpl-gallery-nogrid', 'bmh', 'classic', 'dark_background', 'fast', 'fivethirtyeight', 'ggplot', 'grayscale', 'seaborn', 'seaborn-bright', 'seaborn-colorblind', 'seaborn-dark', 'seaborn-dark-palette', 'seaborn-darkgrid', 'seaborn-deep', 'seaborn-muted', 'seaborn-notebook', 'seaborn-paper', 'seaborn-pastel', 'seaborn-poster', 'seaborn-talk', 'seaborn-ticks', 'seaborn-white', 'seaborn-whitegrid', 'tableau-colorblind10']

# visualization library
#sns.set_style(style="bright", color_codes=True)
# sns.set_context(rc={"font.family": 'sans',
#                     "font.size": 12,
#                     "axes.titlesize": 25,
#                     "axes.labelsize": 24,
#                     "ytick.labelsize": 20,
#                     "xtick.labelsize": 20,
#                     "lines.linewidth": 4,
#                     })

def ci2(mean, std, n, conf=0.025):
    # Calculate the t-value
    t_value = t.ppf(1 - conf, n - 1)

    # Calculate the margin of error
    margin_error = t_value * std / math.sqrt(n)

    # Calculate the lower and upper bounds of the confidence interval
    lower_bound = mean - margin_error
    upper_bound = mean + margin_error
    return lower_bound, upper_bound

np.random.seed(200)


if __name__ == '__main__':
    # algorithms_to_use = ['sliding_w_UCB', 'epsilon_greedy', 'exp3S',
    #                      'our_policy']
    if os.getcwd().endswith("plots"):
        os.chdir("..")

    print(os.getcwd())
    save_files = True

    base_path_first = ('ICML_experiments_error/3states_4actions_5obs/pomdp10/estimation_error')

    fig, axs = plt.subplots(1, 2, figsize=(20, 6))  # , sharex=True, sharey=True)
    # plot_titles = ['(a)', '(b)']

    # first exp
    num_checkpoints = 200
    checkpoint_size = 1000      # number of couples
    basic_info_path = base_path_first + f"/{checkpoint_size}_{num_checkpoints}cp_0.json"
    f = open(basic_info_path)
    data = json.load(f)
    f.close()
    frobenious_norm_first = np.array(data["error_frobenious_norm"])

    print(frobenious_norm_first.shape)

    mean_frobenious_first = frobenious_norm_first.mean(axis=0)
    std_frobenious_first = frobenious_norm_first.std(axis=0)
    lower_bound_first, upper_bound_first = ci2(mean_frobenious_first,
                                   std_frobenious_first, frobenious_norm_first.shape[0])

    x_axis = np.array([checkpoint_size*(i+1) for i in range(num_checkpoints)])
    x_axis_mask = np.array([i % 10 == 0 for i in range(num_checkpoints)])
    x_axis_mask[0] = False

    axs[0].plot(x_axis[x_axis_mask], mean_frobenious_first[x_axis_mask], 'c', label='3stat 4act 5obs')
    axs[0].fill_between(x_axis[x_axis_mask],
                        lower_bound_first[x_axis_mask],
                        upper_bound_first[x_axis_mask],
                        color='c', alpha=.2)

    first_result_dict = \
        {'x_axis_first': x_axis[x_axis_mask] / 10**5,
         'mean_frobenious_first': mean_frobenious_first[x_axis_mask],
         'lower_bound_first': lower_bound_first[x_axis_mask],
         'upper_bound_first': upper_bound_first[x_axis_mask],
        }

    # second exp
    base_path_second = (
        'ICML_experiments_error/5states_3actions_8obs/pomdp4/estimation_error')
    num_checkpoints = 200
    checkpoint_size = 1000  # number of couples
    basic_info_path = base_path_second + f"/{checkpoint_size}_{num_checkpoints}cp_0.json"
    f = open(basic_info_path)
    data = json.load(f)
    f.close()
    frobenious_norm_second = np.array(data["error_frobenious_norm"])

    print(frobenious_norm_second.shape)

    mean_frobenious_second = frobenious_norm_second.mean(axis=0)
    std_frobenious_second = frobenious_norm_second.std(axis=0)
    lower_bound_second, upper_bound_second = ci2(mean_frobenious_second,
                                               std_frobenious_second,
                                               frobenious_norm_second.shape[0])

    x_axis = np.array(
        [checkpoint_size * (i + 1) for i in range(num_checkpoints)])
    x_axis_mask = np.array([i % 10 == 0 for i in range(num_checkpoints)])
    x_axis_mask[0] = False

    axs[0].plot(x_axis[x_axis_mask], mean_frobenious_second[x_axis_mask], 'g',
             label='5stat 3act 8obs')
    axs[0].fill_between(x_axis[x_axis_mask],
                        lower_bound_second[x_axis_mask],
                        upper_bound_second[x_axis_mask],
                        color='g', alpha=.2)

    axs[0].legend(fontsize=20)

    first_result_dict['x_axis_second'] = x_axis[x_axis_mask] / 10**5
    first_result_dict['mean_frobenious_second'] = mean_frobenious_second[x_axis_mask]
    first_result_dict['lower_bound_second'] = lower_bound_second[x_axis_mask]
    first_result_dict['upper_bound_second'] = upper_bound_second[x_axis_mask]

    first_df = pd.DataFrame(first_result_dict)

    if save_files:
        first_df.to_csv('ICML_experiments_files/left_figure.csv', index=False)
    # create pandas and then csv
    # pd.DataFrame()


    base_path_third = (
        'ICML_experiments_error/8states_5actions_15obs/pomdp4/estimation_error')

    # third exp
    num_checkpoints = 40
    checkpoint_size = 250000      # number of couples
    basic_info_path = base_path_third + f"/{checkpoint_size}_{num_checkpoints}cp_0_first.json"
    f = open(basic_info_path)
    data = json.load(f)
    f.close()
    frobenious_norm_third = np.array(data["error_frobenious_norm"])

    second_base_info_path = base_path_third + f"/{checkpoint_size}_{num_checkpoints}cp_1.json"
    f = open(second_base_info_path)
    data = json.load(f)
    f.close()

    frobenious_norm_third = np.vstack([frobenious_norm_third, np.array(data["error_frobenious_norm"])])

    print(frobenious_norm_third.shape)

    mean_frobenious_third = frobenious_norm_third.mean(axis=0)
    std_frobenious_third = frobenious_norm_third.std(axis=0)
    lower_bound_third, upper_bound_third = ci2(mean_frobenious_third,
                                   std_frobenious_third, frobenious_norm_third.shape[0])

    x_axis = np.array([checkpoint_size*(i+1) for i in range(num_checkpoints)])
    x_axis_mask = np.array([i > 0 for i in range(num_checkpoints)])

    axs[1].plot(x_axis[x_axis_mask], mean_frobenious_third[x_axis_mask], 'r', label='8stat 5act 15obs')
    axs[1].fill_between(x_axis[x_axis_mask],
                        lower_bound_third[x_axis_mask],
                        upper_bound_third[x_axis_mask],
                        color='r', alpha=.2)

    second_result_dict = \
        {'x_axis_third': x_axis[x_axis_mask] / 10**7,
         'mean_frobenious_third': mean_frobenious_third[x_axis_mask],
         'lower_bound_third': lower_bound_third[x_axis_mask],
         'upper_bound_third': upper_bound_third[x_axis_mask],
        }

    # fourth exp
    base_path_fourth = (
        'ICML_experiments_error/10states_6actions_15obs/pomdp2/estimation_error')
    num_checkpoints = 40
    checkpoint_size = 250000  # number of couples
    basic_info_path = base_path_fourth + f"/{checkpoint_size}_{num_checkpoints}cp_0.json"
    f = open(basic_info_path)
    data = json.load(f)
    f.close()
    frobenious_norm_fourth = np.array(data["error_frobenious_norm"])

    second_base_info_path = base_path_third + f"/{checkpoint_size}_{num_checkpoints}cp_1.json"
    f = open(second_base_info_path)
    data = json.load(f)
    f.close()

    frobenious_norm_fourth = np.vstack(
        [frobenious_norm_fourth, np.array(data["error_frobenious_norm"])])

    print(frobenious_norm_fourth.shape)

    mean_frobenious_fourth = frobenious_norm_fourth.mean(axis=0)
    std_frobenious_fourth = frobenious_norm_fourth.std(axis=0)
    lower_bound_fourth, upper_bound_fourth = ci2(mean_frobenious_fourth,
                                               std_frobenious_fourth,
                                               frobenious_norm_fourth.shape[0] * 3)

    x_axis = np.array(
        [checkpoint_size * (i + 1) for i in range(num_checkpoints)])
    x_axis_mask = np.array([i > 0 for i in range(num_checkpoints)])

    axs[1].plot(x_axis[x_axis_mask], mean_frobenious_fourth[x_axis_mask], 'b',
             label='10stat 6act 15obs')
    axs[1].fill_between(x_axis[x_axis_mask],
                        lower_bound_fourth[x_axis_mask],
                        upper_bound_fourth[x_axis_mask],
                        color='b', alpha=.2)

    axs[1].legend(fontsize=20)

    second_result_dict['x_axis_fourth'] = x_axis[x_axis_mask] / 10**7
    second_result_dict['mean_frobenious_fourth'] = mean_frobenious_fourth[x_axis_mask]
    second_result_dict['lower_bound_fourth'] = lower_bound_fourth[x_axis_mask]
    second_result_dict['upper_bound_fourth'] = upper_bound_fourth[x_axis_mask]

    second_df = pd.DataFrame(second_result_dict)

    if save_files:
        second_df.to_csv('ICML_experiments_files/right_figure.csv', index=False)

    plt.tight_layout()
    plt.show()