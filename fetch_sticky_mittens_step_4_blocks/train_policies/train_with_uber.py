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

env = gym.make("FetchStack4Stage3-v1")

obs = env.reset()
state_length = len(obs['observation']) + len(obs['achieved_goal'])

print("State length - ", state_length)

uber_count = 5

basic_actions_length = env.action_space.shape[0]
action_length = uber_count
exp_length = 100
saving_freq = 20
reward_sum = 0.0

fetch_policy = policies.Policy(state_length, action_length, \
actor_file = "../networks/actor_network.pkl", critic_file = "../networks/critic_network.pkl")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# fetch_policy.save_to_disk()


t = 0
experience_collection = {}
for episode_index in range(1, 2000000):

    obs = env.reset()
    exploration_number = np.exp(-1 * 1 * 1e-3 * (episode_index+1)) # 0.2
    done = False

    reward = 0.0
    step_info = {}
    trajectory = {}
    sub_obs = {}
    action = 0
    state_goal = 0
    option_number = -1
    execution_stack = []


    episode_time = 0
    learning_time = 0

    option_time_count = 0

    while not done:
        env.render()

        # If action being called is available, implement it or call null action otherwise

        if not any(execution_stack):

            current_state = obs['observation']
            current_state = current_state.astype(np.float32)
            current_goal = obs['desired_goal']
            current_goal = current_goal.astype(np.float32)
            achieved_goal = obs['achieved_goal']
            achieved_goal = achieved_goal.astype(np.float32)


            current_state_goal = torch.cat((torch.tensor(current_state), torch.tensor(current_goal)))
            torch_current_state = torch.tensor(current_state)

            if(np.random.uniform(0,1) <= exploration_number):
                action = np.random.randint(action_length-1)
                python_action = action
                # print("Picking random action - ", python_action)
            else:
                # print("input - ", current_state_goal)
                current_state_goal = current_state_goal.to(device)
                torch_distribution = fetch_policy.critic_net.forward(current_state_goal)
                # print("output - ", torch_distribution)
                val, action = torch_distribution.max(0)
                # print("Action - ", action)
                python_action = action.item()
                # print("Picking network action - ", python_action)

            # Critic Value
            # value = fetch_policy.critic_net.forward(current_state_goal)

            # Form the sub policy to be called, push to the stack an

            if python_action == (uber_count - 1):
                current_option = action_policies.do_nothing()
            else:
                current_option = action_policies.place_object("object_" + str(python_action))

            if current_option.initial_predicate(obs):
                execution_stack.append(current_option)
            else:
                current_option = action_policies.do_nothing()
                execution_stack.append(current_option)

            # print("Action - ", action)
            if any(execution_stack):
                sub_obs = copy.deepcopy(obs)
                continue
        else :
            option_time_count += 1
            option_action = execution_stack[-1].compute_option_policy(sub_obs)


            sub_obs, reward, done, info = env.step(option_action)

            reward = round( (reward + (uber_count-1))/(uber_count-1) , 3)

            # print("OPtion time - ", option_time_count)
            if(execution_stack[-1].termination_condition(sub_obs)) or (option_time_count > 50):

                option_time_count = 0
                # print("Exiting current option")
                execution_stack.pop()

                next_state = sub_obs['observation'].astype(np.float32)
                current_goal = sub_obs['desired_goal'].astype(np.float32)
                torch_next_state = torch.tensor(next_state)
                next_state_goal = torch.cat((torch.tensor(next_state), torch.tensor(current_goal))).to(device)

                step_info = {}

                step_info['current_state'] = torch_current_state
                step_info['current_state_goal'] = current_state_goal.to(device)
                step_info['action'] = python_action
                # step_info['action_log'] = log_prob
                step_info['reward'] = reward
                # step_info['QValue'] = value
                step_info['next_state_goal'] = next_state_goal
                step_info['goal'] = current_goal
                step_info['achieved_goal'] = achieved_goal

                if reward > 0.99 :
                    step_info['terminal'] = True
                else:
                    step_info['terminal'] = False

                trajectory[t] = step_info
                experience_collection[t % exp_length] = copy.deepcopy(step_info)
                t = t + 1

                obs = copy.deepcopy(sub_obs)

                fetch_policy.update_policy(experience_collection)

    reward_sum += reward


    print("At episode ", episode_index, " end reward - ", reward)

    if((episode_index % saving_freq) == 0):
        # When done, update the networks with the experience you just picked up
        fetch_policy.save_to_disk()
        avg_reward = reward_sum / saving_freq
        reward_sum = 0.0
        print("At episode - ", episode_index, "\t Average reward - ", avg_reward \
        , " exploration_number - ", exploration_number)


env.close()
