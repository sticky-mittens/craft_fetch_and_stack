from __future__ import division
from __future__ import print_function

import numpy as np
import time
import sys
import os
import json

import torch
import env_factory
from output import policies
import learn_sub_task_no_track

def run_till_call(env, main_thing_name, current_thing_name, visualise = False, model_folder = "./models"):


    actor_network_file = model_folder + "/actor_" + str(main_thing_name) + "_network.pkl"
    critic_network_file = model_folder + "/critic_" + str(main_thing_name) + "_network.pkl"
    policy_info_file = model_folder + "/policy_info_" + str(main_thing_name) + ".json"

    policy_info = {"actor" : actor_network_file, "critic" : critic_network_file, \
      "info": policy_info_file, "termination" : main_thing_name}


    basic_actions = env.action_specs()
    state = env.obs_specs()
    state_length = state['features']['shape'][0]

    execution_stack = []
    execution_stack.append(policy_info)

    # Find the actions corresponding to function calls
    actions = {}
    with open(execution_stack[-1]["info"], "r") as fp:
        actions = json.load(fp)

    execution_stack[-1]["action_info"] = actions
    policy = policies.Policy(state_length, 1, len(actions), \
    actor_file = execution_stack[-1]["actor"], critic_file = execution_stack[-1]["critic"] )
    execution_stack[-1]["policy"] = policy

    state = env.observations()

    t = 0
    reward = 0.0
    done = False

    while any(execution_stack) and (not done):

        # print("Highest level goal - ", main_thing_name, " looking to reset to - ", current_thing_name, " Current mode - ", execution_stack[-1]["termination"])

        if env._sub_task_accomplished(current_thing_name):
            # Found the item you are trying to generate call to
            execution_stack = []
            return env

        # observe state
        current_state = state['features']

        # Pick the policy from the last member in the execution stack

        current_policy = execution_stack[-1]["policy"]

        torch_distribution = current_policy.actor_net.forward(torch.tensor(current_state))
        distribution = torch_distribution.clone().detach().numpy()

        action = np.random.choice(len(execution_stack[-1]["action_info"]), p = np.squeeze(distribution))

        # If the action is in ORDINARY actions then execute it in the environment and continue
        if action in list(basic_actions.values()):
            reward, done, state = env.step(action)
        else:
        # else load the policy for the uber action taken, and push it to the execution stack
            action_name = list(execution_stack[-1]["action_info"].keys())\
            [list(execution_stack[-1]["action_info"].values()).index(action)]

            thing_name = learn_sub_task_no_track.get_thing_from_uber(action_name)

            # print("Executing uber action for thing - ", thing_name)

            if thing_name == current_thing_name or  env._sub_task_accomplished(current_thing_name):
                # Found the item you are trying to generate call to
                execution_stack = []
                return env


            if execution_stack[-1]["termination"] != thing_name and \
            (not env._sub_task_accomplished(thing_name)) :

                # print("Calling uber action - ", action_name)

                actor_network_file = model_folder + "/actor_" + thing_name + "_network.pkl"
                critic_network_file = model_folder + "/critic_" + thing_name + "_network.pkl"
                policy_info_file = model_folder + "/policy_info_" + thing_name + ".json"

                if os.path.exists(actor_network_file) and os.path.exists(critic_network_file) and os.path.exists(policy_info_file) :

                    current_program = {"actor" : actor_network_file, "critic" : critic_network_file, \
                    "info": policy_info_file, "termination" : thing_name}

                    next_actions = {}
                    with open(policy_info_file, "r") as fp:
                        next_actions = json.load(fp)

                    current_program["action_info"] = next_actions

                    policy = policies.Policy(state_length, 1, len(next_actions), \
                    actor_file = current_program["actor"],\
                    critic_file = current_program["critic"])

                    current_program["policy"] = policy

                    execution_stack.append(current_program)
                else:

                    reward, done, state = env.step_sub_task(action_name, thing_name)


        # If the current goal is reached, remove it from the execution stack
        if env._sub_task_accomplished(execution_stack[-1]["termination"]) :
            ret = execution_stack.pop()


        t += 1

        if visualise:
          env.render_matplotlib(frame=state['image'])

    return env