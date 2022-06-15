from __future__ import division
from __future__ import print_function

import numpy as np
import time
import sys
import json

import torch
import env_factory
from output import policies_finetune as policies
import generate_calls

def learn(env, real_goal_name, current_goal_name, episodes, actions_available, visualise = False, model_folder = "./models"):

    real_thing = get_thing_name(real_goal_name)
    goal_thing = get_thing_name(current_goal_name)
    actions = actions_available[0]
    basic_actions = actions_available[1]
    uber_actions = actions_available[2]

    calls_made = {}

    state = env.obs_specs()
    state_length = state['features']['shape'][0]

    craft_policy = policies.Policy(state_length, 1, len(actions), \
    actor_file = model_folder + "/actor_" + goal_thing + "_" + "network.pkl",\
    critic_file = model_folder + "/critic_" + goal_thing + "_" + "network.pkl")

    filename = model_folder + "/policy_info_" + goal_thing + ".json"
    with open(filename, "w") as fp :
        json.dump(actions_available[0], fp, indent = 1)

    reward_sum = 0.0
    avg_reward = 0.0
    learning_freq = 20
    experience_collection = {}
    for episode_index in range(1, episodes):


        exploration_number = np.exp(-5 * 1e-3 * (episode_index+1)) # OG: np.exp(-5 * 1e-3 * (episode_index+1))
        # print("Starting episode : ", episode_index)
        # For each episode you run till the goal is achived
        # or the you run out of maximum  number of steps allowed
        done = False
        state = env.reset()

        # if any(initial_positions):
        #     init_pos_index = episode_index % len(initial_positions)
        #     init_pos = initial_positions[init_pos_index]
        #     env.reset_position(init_pos)

        if real_goal_name != current_goal_name :
            # print("Setting out to reset to an appropriate call position ........... ")
            env = generate_calls.run_till_call(env, real_thing, goal_thing, visualise, model_folder)
            state = env.observations()
            # print("Returned from the appropriate call position.")


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
            state = []
            step_info = {}
            step_info['is_uber'] = False

            if action in basic_actions.values():
                reward, done, state = env.step_sub_task(action, goal_thing)
            else:
                action_name = list(uber_actions.keys())[list(uber_actions.values()).index(action)]
                reward, done, state = env.step_sub_task(action_name, goal_thing)

                if action_name in uber_actions.keys() :
                    uber_count += 1

                    thing = get_thing_from_uber(action_name)

                    if env._sub_task_accomplished(thing) :
                        step_info['is_uber'] = True
                        if thing in calls_made.keys():
                            calls_made[thing].append(prev_pos)
                        else:
                            calls_made[thing] = [prev_pos]

                        step_info['uber_thing'] = thing ######### new

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



def get_thing_name(goal_name):
    id1 = goal_name.find("[")
    id2 = goal_name.find("]")
    thing_name = goal_name[id1+1:id2]
    return thing_name


def get_thing_from_uber(action_name):
    id1 = action_name.find("give_") + len("give_")
    thing_name = action_name[id1:]
    return thing_name
