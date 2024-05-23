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

def ci2(mean, std, n, conf=0.7):
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

    pompd_num = 16
    base_paths = [f'ICML_experiments/3states_3actions_3obs/pomdp{pompd_num}/regret/'
                         '0.02stst_0.05_minac_primo_buono',
                  f'ICML_experiments/3states_3actions_3obs/pomdp{pompd_num}/regret/'
                  '0.02stst_0.05_minac_3',
                  f'ICML_experiments/3states_3actions_3obs/pomdp{pompd_num}/regret/'
                  '0.02stst_0.05_minac_4',
                  ]

    num_experiments = np.array([3, 3, 4])
    num_total_experiments = num_experiments.sum()
    num_episodes = 9

    consider_opt = True
    consider_spectral = True
    consider_psrl = True

    fig, axs = plt.subplots(1, 1, figsize=(20, 6))  # , sharex=True, sharey=True)
    # plot_titles = ['(a)', '(b)']

    oracle_collected_samples = None
    optimistic_collected_samples = None
    spectral_collected_samples = None
    psrl_collected_samples = None

    for i, base_path in enumerate(base_paths):

        basic_info_path = base_path + "/basic_info.json"
        # oracle_opt_info_path = base_path + "/150_init"
        # spect_info_path = base_path + "/1500_3000_init"
        # psrl_info_path = base_path + "/150_init"

        oracle_opt_info_path = base_path + "/1600_init"
        spect_info_path = base_path + "/20000_65000_init"
        psrl_info_path = base_path + "/1600_init"

        for experiment_num in range(num_experiments[i]):
            current_exp_oracle_data = None
            current_exp_optimistic_data = None
            current_exp_spectral_data = None
            current_exp_psrl_data = None
            for episode_num in range(num_episodes):
                print(f"Experiment {experiment_num} and episode {episode_num}")

                # oracle
                f = open(oracle_opt_info_path + f'/oracle_{episode_num}Ep_{experiment_num}Exp.json')
                data = json.load(f)
                f.close()
                current_oracle_collected_samples = np.array(data["collected_samples"])

                if current_exp_oracle_data is None:
                    current_exp_oracle_data = current_oracle_collected_samples
                else:
                    current_exp_oracle_data = np.vstack([current_exp_oracle_data, current_oracle_collected_samples])


                # # optimistic algorithm
                if consider_opt:
                    f = open(oracle_opt_info_path + f'/optimistic_{episode_num}Ep_{experiment_num}Exp.json')
                    data = json.load(f)
                    f.close()
                    current_optimistic_collected_samples = np.array(data["collected_samples"])

                    if current_exp_optimistic_data is None:
                        current_exp_optimistic_data = current_optimistic_collected_samples
                    else:
                        current_exp_optimistic_data = np.vstack([current_exp_optimistic_data,
                                                              current_optimistic_collected_samples])

                # spectral algorithm
                if consider_spectral:
                    f = open(spect_info_path + f'/spectral_{episode_num}Ep_{experiment_num}Exp.json')
                    data = json.load(f)
                    f.close()
                    current_spectral_collected_samples = np.array(data["collected_samples"])

                    if current_exp_spectral_data is None:
                        current_exp_spectral_data = current_spectral_collected_samples
                    else:
                        current_exp_spectral_data = np.vstack([current_exp_spectral_data,
                                                              current_spectral_collected_samples])


            if consider_psrl:
                f = open(psrl_info_path + f'/psrl_{experiment_num}Exp.json')
                data = json.load(f)
                f.close()
                current_exp_psrl_data = np.array(data["collected_samples"])

            if oracle_collected_samples is None:
                oracle_collected_samples = current_exp_oracle_data[:, 2].reshape(-1, 1)
            else:
                oracle_collected_samples = np.hstack(
                    [oracle_collected_samples, current_exp_oracle_data[:, 2].reshape(-1, 1)])

            if consider_opt:
                if optimistic_collected_samples is None:
                    optimistic_collected_samples = current_exp_optimistic_data[:, 2].reshape(-1, 1)
                else:
                    optimistic_collected_samples = np.hstack(
                        [optimistic_collected_samples, current_exp_optimistic_data[:, 2].reshape(-1, 1)])

            if consider_spectral:
                if spectral_collected_samples is None:
                    spectral_collected_samples = current_exp_spectral_data[:, 2].reshape(-1, 1)
                else:
                    spectral_collected_samples = np.hstack(
                        [spectral_collected_samples, current_exp_spectral_data[:, 2].reshape(-1, 1)])

            if consider_psrl:
                if psrl_collected_samples is None:
                    psrl_collected_samples = current_exp_psrl_data[:, 2].reshape(-1, 1)
                else:
                    if psrl_collected_samples.shape[0] > current_exp_psrl_data.shape[0]:
                        psrl_collected_samples = psrl_collected_samples[:current_exp_psrl_data.shape[0], :]
                    psrl_collected_samples = np.hstack(
                        [psrl_collected_samples, current_exp_psrl_data[:, 2].reshape(-1, 1)])

    # print(f"Size of psrl is {psrl_collected_samples.shape[0]}")
    print(f"Size of oracle is {oracle_collected_samples.shape[0]}")
    print(f"Size of optimistic is {optimistic_collected_samples.shape[0]}")

    min_num_samples = np.min([optimistic_collected_samples.shape[0],
                              oracle_collected_samples.shape[0],
                              #spectral_collected_samples.shape[0]
                              ])

    # print(f"PSRL dataset size is {psrl_collected_samples.shape[0]}")
    print(f"Min dataset size is {min_num_samples}")

    # min_num_samples = np.min([spectral_collected_samples.shape[0], oracle_collected_samples.shape[0]])
    # print(f"Spectral dataset size is {spectral_collected_samples.shape[0]}")

    oracle_collected_samples = oracle_collected_samples[:min_num_samples]
    x_axis = np.array([i for i in range(oracle_collected_samples.shape[0])])

    x_axis_mask = np.array([i % 50000 == 0 for i in range(int(min_num_samples))])


    if consider_opt:
        optimistic_collected_samples = optimistic_collected_samples[:min_num_samples]
        print(
            f'Average of Opt collected samples is {optimistic_collected_samples.mean()}')
        optimistic_regret = oracle_collected_samples - optimistic_collected_samples
        optimistic_regret = optimistic_regret.T

        cumulative_optimistic_regret = np.cumsum(optimistic_regret, axis=1)
        mean_cumulated_optimistic_regret = np.mean(cumulative_optimistic_regret, axis=0)
        std_cumulative_optimistic_regret = np.std(cumulative_optimistic_regret, axis=0)
        lower_bound_opt, upper_bound_opt = ci2(mean_cumulated_optimistic_regret,
                                       std_cumulative_optimistic_regret, 2)

        axs.plot(mean_cumulated_optimistic_regret, 'c', label='OAS-UCRL')
        axs.fill_between(x_axis,
                            lower_bound_opt,
                            upper_bound_opt,
                            color='c', alpha=.2)


    if consider_spectral:
        spectral_collected_samples = spectral_collected_samples[:min_num_samples]
        print(
            f'Average of Spectral collected samples is {spectral_collected_samples.mean()}')
        spectral_regret = oracle_collected_samples - spectral_collected_samples
        spectral_regret = spectral_regret.T

        cumulative_spectral_regret = np.cumsum(spectral_regret, axis=1)
        mean_cumulated_spectral_regret = np.mean(cumulative_spectral_regret, axis=0)
        std_cumulative_spectral_regret = np.std(cumulative_spectral_regret, axis=0)
        lower_bound_spec, upper_bound_spec = ci2(mean_cumulated_spectral_regret,
                                       std_cumulative_spectral_regret, 2)

        axs.plot(mean_cumulated_spectral_regret, 'r', label='SEEU')
        axs.fill_between(x_axis,
                            lower_bound_spec,
                            upper_bound_spec,
                            color='r', alpha=.2)


    if consider_psrl:
        psrl_collected_samples = psrl_collected_samples[:min_num_samples]
        print(f'Average of PSRL collected samples is {psrl_collected_samples.mean()}')
        psrl_regret = oracle_collected_samples - psrl_collected_samples
        psrl_regret = psrl_regret.T

        cumulative_psrl_regret = np.cumsum(psrl_regret, axis=1)
        mean_cumulated_psrl_regret = np.mean(cumulative_psrl_regret, axis=0)
        std_cumulative_psrl_regret = np.std(cumulative_psrl_regret, axis=0)
        lower_bound_psrl, upper_bound_psrl = ci2(mean_cumulated_psrl_regret,
                                       std_cumulative_psrl_regret, 2)

        axs.plot(mean_cumulated_psrl_regret, 'g', label='PSRL')
        axs.fill_between(x_axis,
                            lower_bound_psrl,
                            upper_bound_psrl,
                            color='g', alpha=.2)

    dict = {
        'x_axis': x_axis[x_axis_mask] / 10**6,
        'mean_optimistic_regret': mean_cumulated_optimistic_regret[x_axis_mask] / 10**5,
        'lower_bound_opt': lower_bound_opt[x_axis_mask] / 10**5,
        'upper_bound_opt': upper_bound_opt[x_axis_mask] / 10**5,
        'mean_spectral_regret': mean_cumulated_spectral_regret[x_axis_mask] / 10**5,
        'lower_bound_spec': lower_bound_spec[x_axis_mask] / 10**5,
        'upper_bound_spec': upper_bound_spec[x_axis_mask] / 10**5,
        'mean_psrl_regret': mean_cumulated_psrl_regret[x_axis_mask] / 10**5,
        'lower_bound_psrl': lower_bound_psrl[x_axis_mask] / 10**5,
        'upper_bound_psrl': upper_bound_psrl[x_axis_mask] / 10**5,
    }

    df = pd.DataFrame(dict)

    if save_files:
        df.to_csv(f'ICML_experiments_files/regret_info_pomdp{pompd_num}.csv', index=False)

    axs.set_title(f'Pomdp_num{pompd_num}')
    axs.legend()
    plt.tight_layout()
    plt.show()


    # exp_data = []
    # for i, path in enumerate(paths_to_read_from):
    #     # Opening JSON file
    #     f = open(path + '/exp_info.json')
    #     # returns JSON object as
    #     # a dictionary
    #     data = json.load(f)
    #     oracle_list = np.array(data['rewards']['oracle'])
    #     sliding_w_UCB_list = np.array(data['rewards']['sliding_w_UCB'])
    #     epsilon_greedy_list = np.array(data['rewards']['epsilon_greedy'])
    #     exp3S_list = np.array(data['rewards']['exp3S'])
    #     our_policy_list = np.array(data['rewards']['our_policy'])
    #
    #     oracle_rewards = oracle_list[:, :, 1]
    #     x_axis = [i for i in range(oracle_rewards.shape[1])]
    #
    #     sliding_w_UCB_regret = np.mean(oracle_rewards - sliding_w_UCB_list[:, :, 1], axis=0)
    #     epsilon_greedy_regret = np.mean(
    #         oracle_rewards - epsilon_greedy_list[:, :, 1], axis=0)
    #     exp3S_regret = np.mean(oracle_rewards - exp3S_list[:, :, 1], axis=0)
    #     our_policy_regret = np.mean(oracle_rewards - our_policy_list[:, :, 1],
    #                                 axis=0)
    #
    #     print(f"sliding_w_UCB regret {sliding_w_UCB_regret.sum()}")
    #     axs[i].plot(np.cumsum(sliding_w_UCB_regret), 'c', label='SW-UCB')
    #
    #     print(f"Epsilon greedy regret {epsilon_greedy_regret.sum()}")
    #     axs[i].plot(np.cumsum(epsilon_greedy_regret), 'b', label='EPS-gr')
    #
    #     print(f"Exp3S regret {exp3S_regret.sum()}")
    #     axs[i].plot(np.cumsum(exp3S_regret), 'g', label='Exp3.S')
    #
    #     print(f"Our policy regret {our_policy_regret.sum()}")
    #     axs[i].plot(np.cumsum(our_policy_regret), 'r', label='Our')
    #     # for row in range(oracle_rewards.shape[0]):
    #     # axs[i].plot(np.cumsum(oracle_rewards - our_policy_list[:, :, 1]), 'r')
    #     # axs[i].fill_between(x_axis, np.cumsum(our_policy_low), np.cumsum(our_policy_high), alpha=0.2)
    #
    #     axs[i].set_title(plot_titles[i])
    #     #axs[i].legend()
    #     #axs[i].xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'{x/10000}'))
    #     #axs[i].xaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
    #     #axs[i].xaxis.set_minor_formatter(ticker.ScalarFormatter(useMathText=True))
    #     #axs[i].spines['left'].set_linewidth(2)
    #     axs[i].spines['left'].set_linewidth(2)
    #     axs[i].spines['left'].set_visible(True)
    #     axs[i].spines['bottom'].set_linewidth(2)
    #     axs[i].spines['top'].set_linewidth(1)
    #     axs[i].spines['right'].set_linewidth(1)
    #
    #     axs[i].spines['left'].set_capstyle('butt')
    #     axs[i].spines['bottom'].set_capstyle('butt')
    #     axs[i].spines['top'].set_capstyle('butt')
    #     axs[i].spines['right'].set_capstyle('butt')
    #     # axs[i].ticklabel_format(useOffset=True)
    #     axs[i].set_xlabel('t')
    #     axs[i].set_ylabel('$\widehat{\mathcal{R}}(t)$')
    #     if i == 1:
    #         axs[i].legend()
    #
    # #lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
    # #lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    # #fig.legend(lines, labels)
    #
    # #fig.legend(*axs[0].get_legend_handles_labels(),
    # #           loc='upper center', ncol=4)
    #
    # plt.tight_layout()
    # plt.show()
