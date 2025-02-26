import json
import os
import time

import numpy as np

import utils

from policies.discretized_belief_based_policy import \
    DiscretizedBeliefBasedPolicy
from strategy import strategy_helper
from strategy.spectral_helper import SpectralHelper


class SpectralAlgorithmStrategy:

    def __init__(self,
                 num_states,
                 num_actions,
                 num_obs,
                 pomdp,
                 ext_v_i_stopping_cond=0.02,
                 epsilon_state=0.2,
                 min_action_prob=0.1,
                 delta=0.1,
                 discretized_belief_states=None,
                 save_path=None,
                 save_path_num=None
                 ):

        self.num_states = num_states
        self.num_actions = num_actions
        self.num_obs = num_obs
        self.pomdp = pomdp
        self.ext_v_i_stopping_cond = ext_v_i_stopping_cond
        self.epsilon_state = epsilon_state
        self.min_action_prob = min_action_prob
        self.delta = delta

        self.save_path = save_path
        self.save_path_num = save_path_num

        if discretized_belief_states is None:
            self.discretized_belief_states = utils.discretize_continuous_space(self.num_states, epsilon=epsilon_state)

        else:
            self.discretized_belief_states = discretized_belief_states

        self.len_discretized_beliefs = self.discretized_belief_states.shape[0]

    # the values of samples to discard and samples per estimate refer to the number of couples,
    #  thus the timesteps need to be doubled
    def run(self, tau_1, tau_2, starting_episode_num, num_episodes, experiment_num, initial_state):
        current_state = initial_state


        # self.estimated_action_state_dist_per_episode = np.zeros(shape=(
        #     num_episodes, self.num_actions, self.num_actions,
        #     self.num_states, self.num_states))

        if starting_episode_num == 0:
            self.init_policy()

        # self.estimated_transition_matrix_per_episode = np.zeros(shape=(
        #     num_episodes, self.num_states, self.num_actions,
        #     self.num_states))
        #
        # self.error_frobenious_norm_per_episode = np.empty(shape=num_episodes)

        spectral_estimations_per_action = []
        for action in range(self.num_actions):
            current_spectral_helper = SpectralHelper(
                num_states=self.num_states,
                num_obs=self.num_obs,
                action_index=action,
                action_transition_matrix=self.pomdp.state_action_transition_matrix[:, action, :],
                action_observation_matrix=self.pomdp.state_action_observation_matrix[:, action, :],
                possible_rewards=self.pomdp.possible_rewards,
                tau_1=tau_1,
                num_episodes=num_episodes,
            )
            spectral_estimations_per_action.append(current_spectral_helper)

        for episode_num in range(num_episodes):

            self.collected_samples = None
            estimated_transition_matrix = np.zeros(shape=(self.num_states, self.num_actions, self.num_states))
            estimated_observation_matrix = np.zeros(shape=(self.num_states, self.num_actions, self.num_obs))

            for action in range(self.num_actions):
                current_spectral_helper = spectral_estimations_per_action[action]

                action_collected_samples, action_T_hat, action_O_hat, last_state = current_spectral_helper.run(
                    current_state=current_state,
                    episode_num=episode_num
                )

                current_state = last_state

                estimated_transition_matrix[:, action, :] = action_T_hat
                estimated_observation_matrix[:, action, :] = action_O_hat

                if self.collected_samples is None:
                    self.collected_samples = action_collected_samples
                else:
                    self.collected_samples = np.vstack([self.collected_samples, action_collected_samples])

            estimated_transition_matrix, frob_tr, frob_obs = (
                self.compute_error_estimated_transition_observation_matrix(
                    estimated_tr=estimated_transition_matrix,
                    estimated_obs=estimated_observation_matrix
                ))

            current_confidence_bound = self.compute_confidence_bound(tau_1=tau_1, episode_num=episode_num)

            # compute optimistic mdp
            # estimated_state_action_reward = self.compute_state_action_reward(estimated_observation_matrix
            #                                                                  )
            # optimistic_transition_matrix_mdp, optimistic_policy_mdp = (
            #     strategy_helper.compute_optimistic_MDP(
            #         num_states=self.num_states,
            #         num_actions=self.num_actions,
            #         min_action_prob=self.min_action_prob,
            #         state_action_transition_matrix=estimated_transition_matrix,
            #         ext_v_i_stopping_cond=self.ext_v_i_stopping_cond,
            #         state_action_reward=estimated_state_action_reward,
            #         confidence_bound=current_confidence_bound,
            #         min_transition_value=self.pomdp.min_transition_value
            #     ))

            optimistic_transition_matrix_mdp, optimistic_policy_mdp = (
                strategy_helper.compute_optimistic_MDP(
                    num_states=self.num_states,
                    num_actions=self.num_actions,
                    min_action_prob=self.min_action_prob,
                    state_action_transition_matrix=estimated_transition_matrix,
                    ext_v_i_stopping_cond=self.ext_v_i_stopping_cond,
                    state_action_reward=self.pomdp.state_action_reward,
                    confidence_bound=current_confidence_bound,
                    min_transition_value=self.pomdp.min_transition_value
                ))

            start_time = time.time()
            # compute belief action belief matrix from the optimistic mdp
            optimistic_belief_action_belief_matrix = (
                strategy_helper.compute_belief_action_belief_matrix(
                    num_actions=self.num_actions,
                    num_obs=self.num_obs,
                    discretized_belief_states=self.discretized_belief_states,
                    len_discretized_beliefs=self.len_discretized_beliefs,
                    state_action_transition_matrix=optimistic_transition_matrix_mdp,
                    state_action_observation_matrix=self.pomdp.state_action_observation_matrix
                    # state_action_observation_matrix=estimated_observation_matrix
                ))
            end_time = time.time()
            compute_belief_action_list_time = end_time - start_time
            print(f"Compute_belief_action_list time is {compute_belief_action_list_time}")

            start_time = time.time()
            optimistic_belief_action_mapping = strategy_helper.compute_optimal_POMDP_policy(
                num_actions=self.num_actions,
                discretized_belief_states=self.discretized_belief_states,
                len_discretized_beliefs=self.len_discretized_beliefs,
                ext_v_i_stopping_cond=self.ext_v_i_stopping_cond,
                min_action_prob=self.min_action_prob,
                # state_action_reward=estimated_state_action_reward,
                state_action_reward=self.pomdp.state_action_reward,
                belief_action_belief_matrix=optimistic_belief_action_belief_matrix
            )
            end_time = time.time()
            optimal_POMDP_policy_time = end_time - start_time
            print(f"Optimal_POMDP_policy_time is {optimal_POMDP_policy_time}")

            self.policy.update_policy_infos(
                state_action_transition_matrix=optimistic_transition_matrix_mdp,
                belief_action_dist_mapping=optimistic_belief_action_mapping
            )

            ###########################
            # self.policy.state_action_observation_matrix = estimated_observation_matrix

            num_last_samples_for_belief_update = 70
            self.policy.update_belief_from_samples(
                action_obs_samples=self.collected_samples[
                                   -num_last_samples_for_belief_update:]
            )

            episode_collected_samples, current_state = self.collect_samples_in_episode(
                starting_state=current_state,
                tau_2=tau_2,
                episode_num=episode_num
            )

            self.collected_samples = np.vstack(
                [self.collected_samples, episode_collected_samples])


            self.save_results(tau_1=tau_1,
                              tau_2=tau_2,
                              episode_num=episode_num,
                              experiment_num=experiment_num,
                              starting_state=current_state,
                              optimistic_belief_action_belief_matrix=optimistic_belief_action_belief_matrix,
                              optimistic_transition_matrix_mdp=optimistic_transition_matrix_mdp,
                              optimistic_belief_action_mapping=optimistic_belief_action_mapping,
                              estimated_transition_matrix=estimated_transition_matrix,
                              frobenious_norm_tr=frob_tr,
                              frobenious_norm_obs=frob_obs,
                              )

        # return (self.collected_samples,
        #         self.estimated_transition_matrix_per_episode,
        #         self.error_frobenious_norm_per_episode)


    def collect_samples_in_episode(self, starting_state, tau_2, episode_num):

        # for convenience these numbers are even
        num_total_samples = int(tau_2 * np.sqrt(episode_num+1))
        first_state = starting_state

        episode_collected_samples = np.zeros(shape=(num_total_samples, 3))

        for sample_num in range(num_total_samples):

            first_action = self.policy.choose_action()
            first_obs = np.random.multinomial(
                n=1, pvals=self.pomdp.state_action_observation_matrix[
                    first_state, first_action],
                size=1)[0].argmax()

            second_state = np.random.multinomial(
                n=1, pvals=self.pomdp.state_action_transition_matrix[
                    first_state, first_action], size=1)[
                0].argmax()

            self.policy.update(first_action, first_obs)
            episode_collected_samples[sample_num] = [first_action, first_obs, self.pomdp.possible_rewards[first_obs]]
            #self.collected_samples.append((first_action, first_obs))

            first_state = second_state

            if sample_num % 5000 == 0:
                print(sample_num)

        return episode_collected_samples, first_state
        # self.policy.update_transition_matrix(
        #     estimated_transition_matrix=estimated_trans_matrix)


    def compute_confidence_bound(self, tau_1, episode_num):
        min_transition_value = self.pomdp.state_action_transition_matrix.min()
        min_action_prob = self.min_action_prob
        # lambda_max = 1 - self.num_states*self.num_actions*min_transition_value*min_action_prob
        # print(lambda_max)

        alpha = self.pomdp.min_svd_reference_matrix

        eps_iota_sq = np.power(min_transition_value * min_action_prob, 3/2)

        # first_term = 4 / (alpha**2 * eps_iota_sq)
        first_term = 1 / np.sqrt(eps_iota_sq)
        second_term = np.sqrt(1 + np.log(np.max((episode_num, 1))**3/self.delta)) / np.sqrt(int(tau_1 // 3)*(episode_num+1))

        # this is the bound in frobenious norm
        confidence_bound = first_term * second_term
        # confidence_bound = second_term
        print(f"Confidence bound is {confidence_bound}")

        # TODO sistemare il fatto che usiamo
        #  la norma frobenious rispetto alla norma 1

        return confidence_bound

    def compute_error_estimated_transition_observation_matrix(self, estimated_tr, estimated_obs):

        # transition matrix
        distance_matrix = np.absolute(self.pomdp.state_action_transition_matrix.reshape(-1) -
            estimated_tr.reshape(-1))
        frobenious_norm_tr = np.sqrt(np.sum(distance_matrix**2))
        probability_matrix_estimation_error = abs(np.sum(distance_matrix))
        print(f"Distance vector norm-1 is {probability_matrix_estimation_error}")
        print(
            f"Distance vector, frobenious norm, is {frobenious_norm_tr}")
        print("Real transition matrix is")
        print(self.pomdp.state_action_transition_matrix)
        print("Estimated transition matrix is")
        print(estimated_tr)

        # observation matrix
        distance_matrix = np.absolute(self.pomdp.state_action_observation_matrix.reshape(-1) -
            estimated_obs.reshape(-1))
        frobenious_norm_obs = np.sqrt(np.sum(distance_matrix**2))
        probability_matrix_estimation_error = abs(np.sum(distance_matrix))
        print(f"Distance vector norm-1 is {probability_matrix_estimation_error}")
        print(
            f"Distance vector, frobenious norm, is {frobenious_norm_obs}")
        print("Real observation matrix is")
        print(self.pomdp.state_action_observation_matrix)
        print("Estimated observation matrix is")
        print(estimated_obs)

        # fix the transition matrix if some negative numbers are present
        if np.any(estimated_tr <= 0):
            modified_transition_matrix = estimated_tr.copy()
            counter = 1
            while np.any(modified_transition_matrix < self.pomdp.min_transition_value - 0.05):
                modified_transition_matrix[
                    modified_transition_matrix <= self.pomdp.min_transition_value] = self.pomdp.min_transition_value + 0.02 * counter
                modified_transition_matrix[
                    modified_transition_matrix >= (
                                1 - self.pomdp.non_normalized_min_transition_value)] = 1 - self.pomdp.non_normalized_min_transition_value
                sum_over_last_state = modified_transition_matrix.sum(axis=2)
                modified_transition_matrix = (modified_transition_matrix /
                                              sum_over_last_state[:, :, None])
                counter += 1
            estimated_tr = modified_transition_matrix

        return estimated_tr, frobenious_norm_tr, frobenious_norm_obs


    def compute_state_action_reward(self, estimated_observation_matrix):
        state_action_reward = np.zeros(shape=(self.num_states, self.num_actions))
        for state in range(self.num_states):
            for action in range(self.num_actions):
                mean_reward = np.sum(
                    self.pomdp.possible_rewards *
                    estimated_observation_matrix[state, action])
                state_action_reward[state, action] = mean_reward

        return state_action_reward


    def save_results(self, tau_1, tau_2, episode_num, experiment_num,
                     starting_state,
                     optimistic_belief_action_belief_matrix,
                     optimistic_transition_matrix_mdp,
                     optimistic_belief_action_mapping,
                     estimated_transition_matrix,
                     frobenious_norm_tr,
                     frobenious_norm_obs,
                     ):

        if not isinstance(self.policy.discretized_belief_index, int):
            index_to_store = self.policy.discretized_belief_index
        else:
            index_to_store = self.policy.discretized_belief_index.tolist()

        if not isinstance(starting_state, int):
            starting_state = starting_state.tolist()

        result_dict = {
            "tau_1": tau_1,
            "tau_2": tau_2,
            # "starting_state": starting_state,
            #"discretized_belief": self.policy.discretized_belief.tolist(),
            #"discretized_belief_index": index_to_store,
            #"optimistic_belief_action_belief_matrix": optimistic_belief_action_belief_matrix,  #.tolist(),
            "optimistic_transition_matrix_mdp": optimistic_transition_matrix_mdp.tolist(),
            "optimistic_belief_action_mapping": optimistic_belief_action_mapping.tolist(),
            "estimated_transition_matrix": estimated_transition_matrix.tolist(),
            "frobenious_norm_tr": frobenious_norm_tr.tolist(),
            "frobenious_norm_obs": frobenious_norm_obs.tolist(),
            "collected_samples": self.collected_samples.tolist()
        }

        basic_info_path = f"/{self.epsilon_state}stst_{self.min_action_prob}_minac_{self.save_path_num}/{tau_1}_{tau_2}_init"
        dir_to_create_path = self.save_path + basic_info_path
        if not os.path.exists(dir_to_create_path):
            os.mkdir(dir_to_create_path)
        f = open(
            dir_to_create_path + f'/spectral_{episode_num}Ep_{experiment_num}Exp.json',
            'w')
        json_file = json.dumps(result_dict)
        f.write(json_file)
        f.close()
        print(f"Optimistic Results of episode {episode_num} and experiment {experiment_num} have been saved")


    def restore_infos(self,
                      loaded_data):
        self.policy = DiscretizedBeliefBasedPolicy(
            num_states=self.num_states,
            num_actions=self.num_actions,
            num_obs=self.num_obs,
            initial_discretized_belief=None,
            initial_discretized_belief_index=None,
            discretized_beliefs=self.discretized_belief_states,
            estimated_state_action_transition_matrix=np.array(loaded_data["optimistic_transition_matrix_mdp"]),
            belief_action_dist_mapping=np.array(loaded_data["optimistic_belief_action_mapping"]),
            state_action_observation_matrix=self.pomdp.state_action_observation_matrix,
            no_info=False
        )

        self.policy.discretized_belief = np.array(loaded_data["discretized_belief"])
        self.policy.discretized_belief_index = loaded_data["discretized_belief_index"]


    def init_policy(self):

        # used policy
        self.policy = DiscretizedBeliefBasedPolicy(
            num_states=self.num_states,
            num_actions=self.num_actions,
            num_obs=self.num_obs,
            initial_discretized_belief=None,
            initial_discretized_belief_index=None,
            discretized_beliefs=self.discretized_belief_states,
            estimated_state_action_transition_matrix=None,
            belief_action_dist_mapping=None,
            state_action_observation_matrix=self.pomdp.state_action_observation_matrix,
            no_info=True
        )

    def generate_basic_info_dict(self):

        experiment_basic_info = {
            "discretized_belief_states": self.discretized_belief_states.tolist(),
            "ext_v_i_stopping_cond": self.ext_v_i_stopping_cond,
            "epsilon_state": self.epsilon_state,
            "min_action_prob": self.min_action_prob
        }

        return experiment_basic_info