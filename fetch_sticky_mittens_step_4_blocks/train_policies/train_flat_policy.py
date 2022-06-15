import sys
sys.path.append("../")

import gym
import gym_fetch_stack
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from uber_policies import action_policies
import flat_policies
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
h_policy = flat_policies.Option_Collection(args.no_of_blocks, args.folder_name, args.demos_file)



''' If score is above 0.9 then try for next level otherwise stay in current level '''
reward_avg = 0
reward_sum = 0
episode_index = 0
current_exploration = 0.0
experience_collection = {}
mb = []
#while reward_avg < task_finished_thresh or current_exploration > 0.05 :
while episode_index < 10000: 
    obs = env.reset()

    done = False
    reward = 0
    h_policy.book_keeping_reset()

    t = 0
    last_reward = 0
    episode_run = {}

    while not done:
        # env.render()
        transition, env_action = h_policy.execute_policy(obs, episode_index)
        if transition :
            transition['reward'] = reward
            if done:
                transition['terminal'] = True
            experience_collection[t % args.exp_length] = copy.deepcopy(transition)
            h_policy.update_policy(experience_collection)

            t += 1
        else :
            obs, reward, done, info = env.step(env_action)
            reward = round( (reward + (args.no_of_blocks))/(args.no_of_blocks) , 3)
            # reward = min(0, reward) + 1


    print("At episode ", episode_index, " end reward - ", reward)


    reward_sum += reward
    episode_index += 1

    # Print some stats and save some of the networks involved in the process

    if((episode_index % args.saving_freq) == 0):
        # When done, update the networks with the experience you just picked up

        h_policy.save_to_disk()

        reward_avg = reward_sum / args.saving_freq
        reward_sum = 0.0
        print("At episode - ", episode_index, "\t Average reward - ", reward_avg )

        current_exploration = np.exp(-1 * h_policy.decay_rate * 1e-3 * \
        (h_policy.episode_count + 1) )
        print("exploration_numbers : ", current_exploration)

env.close()
