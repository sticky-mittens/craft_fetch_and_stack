from __future__ import division
from __future__ import print_function

import numpy as np
import time
import sys
import os, json

import torch
import env_factory
from output import policies_finetune as policies
DEBUG = False

def learn(env, goal_name, episodes, actions_available, uid, visualise = False):

    goal_thing = get_thing_name(goal_name)
    actions = actions_available[0]
    basic_actions = actions_available[1]
    uber_actions = actions_available[2]


    calls_made = {}

    state = env.obs_specs()
    state_length = state['features']['shape'][0]

    craft_policy = policies.Policy(state_length, 1, len(actions), \
    actor_file = "models/"+uid+"/actor_" + goal_thing + "_" + "network.pkl",\
    critic_file = "models/"+uid+"/critic_" + goal_thing + "_" + "network.pkl")
    filename = "models/"+uid+"/policy_info_" + goal_thing + ".json"
    with open(filename, "w") as fp :
        json.dump(actions_available[0], fp, indent = 1)

    reward_sum = 0.0
    avg_reward = 0.0
    learning_freq = 20
    experience_collection = {}
    for episode_index in range(1, episodes):


        exploration_number = np.exp(-5 * 1e-3 * (episode_index+1))
        # print("Starting episode : ", episode_index)
        # For each episode you run till the goal is achived
        # or the you run out of maximum  number of steps allowed
        done = False
        state = env.reset()

        # reset to random position
        env.random_reset_position()
            

        experience = {}
        uber_count = 0
        t = 0
        while(not done):
            current_state = state['features']


            action = 0
            log_prob = 0
            # Pick an action according to the actor :

            if(np.random.uniform(0,1) <= exploration_number):
                action = np.random.randint(len(actions))
                torch_distribution = craft_policy.actor_net.forward(torch.tensor(current_state))
                log_prob = torch.log(torch_distribution[action])
            else:
                torch_distribution = craft_policy.actor_net.forward(torch.tensor(current_state))
                distribution = torch_distribution.clone().detach().numpy()
                action = np.random.choice(len(actions), p = np.squeeze(distribution))
                log_prob = torch.log(torch_distribution[action])


            # Critic Value
            value = craft_policy.critic_net.forward(torch.tensor(current_state))

            prev_pos = env._current_state.pos

            reward = 0
            done = False
            step_info = {}
            step_info['is_uber'] = False

            if action in basic_actions.values():
                reward, done, state = env.step_sub_task(action, goal_thing)
                step_info['steps_since_last_learning_action'] = 1
            else: # if uber action, then
                reward, done, state, steps_for_option_execution = execute_option(state, state_length, action, basic_actions, actions, reward, done, env, goal_thing, visualise, uid)
                step_info['steps_since_last_learning_action'] = steps_for_option_execution # new for bottom up: 
            
            step_info['state'] = current_state
            step_info['action'] = action
            step_info['action_log'] = log_prob
            step_info['reward'] = reward
            step_info['QValue'] = value

            experience[t] = step_info

            if visualise:
              env.render_matplotlib(frame=state['image'])

            if reward:

                reward_sum += reward
                if visualise:
                  rewarding_frame = state['image'].copy()
                  rewarding_frame[:40] *= np.array([0, 1, 0])
                  env.render_matplotlib(frame=rewarding_frame, delta_time=0.7)
                # print("[{}] Got a rewaaaard! {:.1f}".format(t, reward) , " Number of uber actions used - ", uber_count)
            elif done:
                if visualise:
                  env.render_matplotlib(
                      frame=np.zeros_like(state['image']), delta_time=0.3)
                # print("[{}] Finished with nothing... Reset".format(t))
            t = t + 1

        experience_collection[episode_index] = experience
        if((episode_index % learning_freq) == 0):
            # When done, update the networks with the experience you just picked up
            craft_policy.update_policy(experience_collection)
            craft_policy.save_to_disk()
            experience_collection = {}
            calls_made = {}
            avg_reward = reward_sum / learning_freq

            print("At episode - ", episode_index, "Exploration_number - {:.1f}".format(exploration_number), \
            "Average reward - {:.3f}".format(avg_reward))
            reward_sum = 0.0



    print("When returning from function avg reward  - ", avg_reward)
    return avg_reward, calls_made, craft_policy

def execute_option(state, state_length, action, basic_actions, all_actions, reward, done, env, goal_thing, visualise, uid):
    action_name = list(all_actions.keys())[list(all_actions.values()).index(action)]
    if DEBUG: print('[DEBUG] Uber action is ', action_name)

    thing_name = action_name.replace("give_","")
    if DEBUG: print('[DEBUG] playing option for thing={}.'.format(thing_name))

    actor_network_file = "./models/"+uid+"/actor_" + thing_name + "_network.pkl"
    critic_network_file = "./models/"+uid+"/critic_" + thing_name + "_network.pkl"
    policy_info_file = "./models/"+uid+"/policy_info_" + thing_name + ".json"

    assert os.path.exists(actor_network_file)
    assert os.path.exists(critic_network_file)
    assert os.path.exists(policy_info_file)

    next_actions = {}
    with open(policy_info_file, "r") as fp:
        next_actions = json.load(fp)

    current_policy = policies.Policy(state_length, 1, len(next_actions), actor_file = actor_network_file, critic_file = critic_network_file)

    steps_for_option_execution = 0
    max_option_steps = 100
    while (steps_for_option_execution != max_option_steps) and (not env._sub_task_accomplished(thing_name)):

        input = state['features']
        torch_distribution = current_policy.actor_net.forward(torch.tensor(input))
        distribution = torch_distribution.clone().detach().numpy()
        action = np.random.choice(len(next_actions), p = np.squeeze(distribution))

        ### check if action from policy is basic or uber; if uber, have to call this function again ### ! ###
        if action in basic_actions.values():
            reward, done, state = env.step_sub_task(action, goal_thing)
            steps_for_option_execution += 1
        else: # if uber action, then
            reward, done, state, steps_for_option_execution_inner = execute_option(state, state_length, action, basic_actions, next_actions, reward, done, env, goal_thing, visualise, uid)
            steps_for_option_execution += steps_for_option_execution_inner

        if visualise:
            env.render_matplotlib(frame=state['image'])
        # if DEBUG: print('[DEBUG] action, position is ', action, self.pos)
    
    if DEBUG: print('[DEBUG] the number of steps for option to execute is {}'.format(steps_for_option_execution))
    if DEBUG: print('[DEBUG] Is thing={} in inventory, a.k.a. did option accomplish sub_task? ANS: {}'.format(thing_name, env._sub_task_accomplished(thing_name)))
    return reward, done, state, steps_for_option_execution

    

def get_thing_name(goal_name):
    id1 = goal_name.find("[")
    id2 = goal_name.find("]")
    thing_name = goal_name[id1+1:id2]
    return thing_name


def get_thing_from_uber(action_name):
    id1 = action_name.find("give_") + len("give_")
    thing_name = action_name[id1:]
    return thing_name
