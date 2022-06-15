import sys
sys.path.append("../")

import gym
import gym_fetch_stack
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from uber_policies import action_policies
import policies
import torch
import time
import copy

import argparse
import os

parser = argparse.ArgumentParser(description='Fetch SMTUA')
parser.add_argument('--no-of-blocks', default=3, type=int, help='number of blocks (default: 3)')
parser.add_argument('--starting-level', default=0, type=int, help='choose from 0-2 (default: 0)')
parser.add_argument('--total-learning-levels', default=3, type=int, help='(default: 3)')
parser.add_argument('--saving-freq', default=20, type=int, help='(default: 20 episodes)')
parser.add_argument('--learning-freq', default=10, type=int, help='after collecting experience for these many episodes, learn (default: 10)')
parser.add_argument('--exp-length', default=100, type=int, help='no of time steps for experience to collect to eventually learn from (default: 500)')
parser.add_argument('--folder-name', default='block_3_uber_level_0_seed_1', type=str, help='Unique id for this run')
parser.add_argument('--seed', default=1, type=int, help='random seed (default: 1)')
parser.add_argument('--demos-file', default='../gen_demonstration_log/demos.p', type=str, help='location of demos file (default: "../gen_demonstration_log/demos.p")')
args = parser.parse_args()


for f in ['../networks', '../networks/'+args.folder_name]:
    if not os.path.exists(f):
        os.mkdir(f)

torch.manual_seed(args.seed)

episodic_memory = {}

task_finished_thresh = 0.85
env = gym.make("FetchStack" + str(args.no_of_blocks) + "Stage3-v1")
obs = env.reset()
state_length = len(obs['observation']) + len(obs['achieved_goal'])


''' Create the policy at multiple levels '''
h_policy = policies.Hierarchical_Policies(args.no_of_blocks, args.total_learning_levels, args.folder_name, args.demos_file)


for current_learning_level in range(args.starting_level, args.total_learning_levels):

    print("======= Learning at level ============ ", current_learning_level)

    ''' Play game according to the current policy level and obtain average score across some 'k' episodes'''

    ''' If score is above 0.9 then try for next level otherwise stay in current level '''
    reward_avg = 0
    reward_sum = 0
    episode_index = 0
    current_exploration = 0.0
    experience_collection = {}
    mb = []
    while reward_avg < task_finished_thresh or current_exploration > 0.05 :
        obs = env.reset()

        done = False
        reward = 0
        h_policy.book_keeping_reset()

        t = 0
        last_reward = 0
        episode_run = {}
        step_ct = 0
        while not done:
            # env.render()
            transition, env_action = h_policy.execute_policy(current_learning_level, obs, episode_index)
            if transition :
                if current_learning_level != 2:
                    if transition["terminal"]:
                        if reward > 0 and current_learning_level == 1 :
                            transition['reward'] = 1
                        else:
                            transition['reward'] = reward
                    else:
                        transition['reward'] = 0

                # print("Transition - ", transition['reward'])

                if current_learning_level == 2:
                    episode_run[t] = copy.deepcopy(transition)

                else:
                    experience_collection[t % args.exp_length] = copy.deepcopy(transition)
                    h_policy.update_policy(experience_collection, current_learning_level)

                t += 1

                if current_learning_level < args.total_learning_levels - 1:
                    continue
                else: # For the final level you need to try something different
                    assert len(env_action) == 4
                    # print("Action begin taken - ", env_action)
                    obs, reward, done, info = env.step(env_action)
                    # reward = round( (reward + (args.no_of_blocks))/(args.no_of_blocks) , 3)
                    reward = round( (min(0, reward) + args.no_of_blocks)/(args.no_of_blocks) , 3)
            else :
                obs, reward, done, info = env.step(env_action)
                step_ct += 1
                # reward = round( (reward + (args.no_of_blocks))/(args.no_of_blocks) , 3)
                reward = round( (min(0, reward) + args.no_of_blocks)/(args.no_of_blocks) , 3)


        print("At episode ", episode_index, " end reward - ", reward)
        #print("Total steps: ", step_ct)

        if current_learning_level == (args.total_learning_levels - 1):
            episodic_memory[episode_index % args.exp_length] = episode_run
            if ((episode_index+1) % args.learning_freq) == 0:
                h_policy.update_continuous_policy(episodic_memory)


        reward_sum += reward
        episode_index += 1

        # Print some stats and save some of the networks involved in the process

        if((episode_index % args.saving_freq) == 0):
            # When done, update the networks with the experience you just picked up

            h_policy.save_to_disk(current_learning_level)

            reward_avg = reward_sum / args.saving_freq
            reward_sum = 0.0
            print("At episode - ", episode_index, "\t Average reward - ", reward_avg )

            if current_learning_level == 0:
                current_exploration = np.exp(-1 * h_policy.decay_rate * 1e-3 * \
                (h_policy.network_wise_episode_count[0] + 1) )
                print("exploration_numbers : ", current_exploration)
            elif current_learning_level == 1 :
                print("exploration_numbers : ", end='')
                current_exploration = 0
                keys = list(h_policy.network_wise_episode_count[1].keys())
                keys.sort()
                for each_network in keys[0:-1]:
                    exploration_number = np.exp(-1 * h_policy.decay_rate * 1e-3 * \
                    (h_policy.network_wise_episode_count[1][each_network] + 1))
                    print(exploration_number, " ", end="")
                    if exploration_number > current_exploration :
                        current_exploration = exploration_number
                print("\n")
            elif current_learning_level == 2 :
                print("exploration_numbers : ", end='')
                current_exploration = 0
                for each_block in h_policy.network_wise_episode_count[2].keys():
                    for each_action in h_policy.network_wise_episode_count[2][each_block].keys():
                        exploration_number = np.exp(-1 * h_policy.decay_rate * 1e-3 * \
                        (h_policy.network_wise_episode_count[2][each_block][each_action] + 1))
                        # print("Episode count - ", h_policy.network_wise_episode_count[2][each_block][each_action])
                        print(exploration_number, " ", end="")
                        if exploration_number > current_exploration :
                            current_exploration = exploration_number
                    # print("\n")
                print("\n")

env.close()
