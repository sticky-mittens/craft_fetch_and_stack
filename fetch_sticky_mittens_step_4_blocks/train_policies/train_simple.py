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

env = gym.make("FetchStack2Stage1-v1")
# env = gym.wrappers.Monitor(env, './media/',video_callable=lambda episode_id: True,force = True)
# env = gym.wrappers.Monitor(env, './media/',force = True)

def policy(observation, desired_goal):
    action = env.action_space.sample()
    return action

obs = env.reset()
state_length = len(obs['observation']) + len(obs['achieved_goal'])
action_length = env.action_space.shape[0]

learning_freq = 1
experience_collection = {}
reward_sum = 0.0

fetch_policy = policies.Policy(state_length, action_length, \
actor_file = "../networks/actor_network.pkl", critic_file = "../networks/critic_network.pkl")

for episode_index in range(1, 100000):

    obs = env.reset()
    exploration_number = np.exp(-1 * 1e-3 * (episode_index+1))
    done = False
    t = 0
    reward = 0.0
    step_info = {}
    trajectory = {}
    while not done:
        env.render()

        current_state = obs['observation']
        current_state = current_state.astype(np.float32)
        current_goal = obs['desired_goal']
        current_goal = current_goal.astype(np.float32)
        achieved_goal = obs['achieved_goal']
        achieved_goal = achieved_goal.astype(np.float32)
        # Choose action
        action = 0
        # numpy_action = 0

        state_goal = torch.cat((torch.tensor(current_state), torch.tensor(current_goal)))
        action = fetch_policy.actor_net.forward(state_goal)
        numpy_action = action.clone().detach().numpy()

        if(np.random.uniform(0,1) <= exploration_number):
            for id in range(action_length):
                numpy_action[id] += ((np.random.rand() - 0.5) * 2.0)

            numpy_action = numpy_action.astype(np.float32)

        # print("Action taken - ", numpy_action)
        obs, reward, done, info = env.step(numpy_action)

        next_state = obs['observation']
        next_state = next_state.astype(np.float32)
        reward = (reward + 2)/2

        step_info['state'] = current_state
        step_info['action'] = numpy_action
        step_info['reward'] = reward
        step_info['next_state'] = next_state
        step_info['goal'] = current_goal
        step_info['achieved_goal'] = achieved_goal

        fetch_policy.add_experience(step_info)
        fetch_policy.update_policy()

        trajectory[t] = step_info
        t = t + 1

        if( reward < 0.99 ):
            for index in range(min(len(trajectory)-1, 2)):
                past_step = trajectory[index].copy()

                new_goal = step_info['achieved_goal']
                past_step['goal'] = new_goal
                past_step['reward'] = 1.0
                fetch_policy.add_experience(past_step)
                fetch_policy.update_policy()

    reward_sum += reward
    print("At episode ", episode_index, " end reward - ", reward)



    if((episode_index % (10 * learning_freq)) == 0):
        # When done, update the networks with the experience you just picked up
        fetch_policy.save_to_disk()
        avg_reward = reward_sum / (10 * learning_freq)
        reward_sum = 0.0
        print("At episode - ", episode_index, "\t Average reward - ", avg_reward )


env.close()
