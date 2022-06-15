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
import sys
import os
import agent

def get_multilevel_policy(block_count = 3, level = 0, folder_name = 'block_3_uber_level_0'):

    env = gym.make("FetchStack" + str(block_count) + "Stage3-v1")
    obs = env.reset()
    state_length = len(obs['observation']) + len(obs['achieved_goal'])
    memory_size = 7e+5 // 50
    batch_size = 256
    actor_lr = 1e-3
    critic_lr = 1e-3
    gamma = 0.98
    tau = 0.05
    k_future = 4


    policy_program = {}

    if level == 0 :

        level_wise_policy = {}
        action_length = block_count + 1
        fetch_policy = policies.Policy(state_length, action_length, \
        actor_file = "../networks/"+folder_name+"/actor_network_" + str(level) + ".pkl", \
        critic_file = "../networks/"+folder_name+"/critic_network_" + str(level) + ".pkl")

        level_wise_policy["policy"] = fetch_policy
        level_wise_policy["learnable"] = True # Decides whether you need to back-propagate on this policy

        pre_post_functions = {}
        option_policies = {}
        for each_block in range(block_count) :
            current_option = action_policies.place_object("object_" + str(each_block))
            model_wise = {}
            model_wise["pre"] = current_option.initial_predicate
            # model_wise["post"] = current_option.accomplished
            model_wise["post"] = current_option.accomplished
            model_wise["action"] = current_option.compute_option_policy_2

            option_policies[each_block] = model_wise

        # This is the NULL policy
        current_option = action_policies.do_nothing()
        model_wise = {}
        model_wise["pre"] = current_option.initial_predicate
        model_wise["post"] = current_option.termination_condition
        model_wise["action"] = current_option.compute_option_policy
        option_policies[block_count] = model_wise


        # level_wise_policy["pre_n_post_condition"] = pre_post_functions
        level_wise_policy["uber_action"] = option_policies
        level_wise_policy["action_space"] = "discrete"
        level_wise_policy["action_length"] = action_length

        policy_program[level] = level_wise_policy

    elif level == 1:

        ''' 1. Call for level 1 '''
        ''' 2. Edit the previous policy as non-learnable'''

        policy_program = get_multilevel_policy(block_count, level-1, folder_name)
        policy_program[level-1]["learnable"] = False

        level_wise_mapping = {}

        no_of_prev_actions = block_count + 1

        policy_collection = {}
        for each_prev_action in range(no_of_prev_actions):
            block_wise_expansion = {}
            action_length = 3
            fetch_policy = policies.Policy(state_length, action_length, \
            actor_file = "../networks/"+folder_name+"/actor_network_" + str(level) + "_" + str(each_prev_action) + ".pkl",
            critic_file = "../networks/"+folder_name+"/critic_network_" + str(level) + "_" + str(each_prev_action) + ".pkl")

            block_wise_expansion["policy"] = fetch_policy
            block_wise_expansion["learnable"] = True
            block_wise_expansion["action_length"] = action_length

            uber_actions = {}
            pre_post_functions = {}

            option_0_details = {}
            option_0 = action_policies.reach_object("object_" + str(each_prev_action))
            option_0_details["pre"] = option_0.predicate
            option_0_details["post"] = option_0.termination
            option_0_details["action"] = option_0.compute_option_policy
            uber_actions[0] = option_0_details

            option_1_details = {}
            option_1 = action_policies.pick_and_reach_goal("object_" + str(each_prev_action))
            option_1_details["pre"] = option_1.predicate
            option_1_details["post"] = option_1.termination
            option_1_details["action"] = option_1.compute_option_policy_2
            uber_actions[1] = option_1_details

            option_2_details = {}
            option_2 = action_policies.release_and_lift("object_" + str(each_prev_action))
            option_2_details["pre"] = option_2.predicate
            option_2_details["post"] = option_2.termination
            option_2_details["action"] = option_2.compute_option_policy
            uber_actions[2] = option_2_details

            option_3_details = {}
            option_3 = action_policies.do_nothing()
            option_3_details["pre"] = option_3.initial_predicate
            option_3_details["post"] = option_3.termination_condition
            option_3_details["action"] = option_3.compute_option_policy
            uber_actions[3] = option_3_details

            block_wise_expansion["uber_action"] = uber_actions

            policy_collection[each_prev_action] = block_wise_expansion

        level_wise_mapping["uber_action"] = policy_collection
        level_wise_mapping["action_space"] = "discrete"
        policy_program[level] = level_wise_mapping

    elif level == 2 :

        ''' 1. Call for level 2 '''
        ''' 2. Edit previous policy to be non-learnable '''
        ''' 3. Turn the switch for conitnuous learning '''

        ''' Note the pre and post conditions for this current level, comes from the previous level'''

        policy_program = get_multilevel_policy(block_count, level-1, folder_name)
        policy_program[level-1]["learnable"] = False

        level_wise_mapping = {}
        no_of_prev_actions = block_count + 1

        for block_no in range(no_of_prev_actions):
            block_wise_expansion = {}

            if block_no == block_count : # Null policy
                block_wise_expansion[0] = policies.ContZeroPolicy(state_length, action_length)
                block_wise_expansion[1] = policies.ContZeroPolicy(state_length, action_length)
                block_wise_expansion[2] = policies.ContZeroPolicy(state_length, action_length)
            else:
                action_length = env.action_space.shape[0]

                # Using this to drive the function to use
                transition = {}
                transition["level_0_action"] = block_no
                transition["level_1_action"] = 0
                transition["next_state_info"] = copy.deepcopy(obs)

                modified_transition = generate_desired_and_achieved_goal(transition)
                state_length = len(modified_transition["next_state_info"]["observation"])
                goal_length = len(modified_transition["desired_goal"])
                action_bounds = [env.action_space.low[0], env.action_space.high[0]]


                block_wise_expansion[0] = agent.Agent(n_states = state_length,
                                           n_actions = action_length,
                                           n_goals = goal_length,
                                           action_bounds = action_bounds,
                                           capacity = memory_size,
                                           action_size = action_length,
                                           batch_size = batch_size,
                                           actor_lr=actor_lr,
                                           critic_lr=critic_lr,
                                           gamma=gamma,
                                           tau=tau,
                                           k_future=k_future,
                                           file_name = "../networks/"+folder_name+"/network_reach_object_" + str(block_no)+ ".pth")

                if (os.path.exists(block_wise_expansion[0].filename)):
                    block_wise_expansion[0].load_weights()
                    print("Picking up exiting network : ", block_wise_expansion[0].filename)

                transition = {}
                transition["level_0_action"] = block_no
                transition["level_1_action"] = 1
                transition["next_state_info"] = copy.deepcopy(obs)

                modified_transition = generate_desired_and_achieved_goal(transition)
                state_length = len(modified_transition["next_state_info"]["observation"])
                goal_length = len(modified_transition["desired_goal"])
                action_bounds = [env.action_space.low[0], env.action_space.high[0]]

                block_wise_expansion[1] = agent.Agent(n_states = state_length,
                                           n_actions = action_length,
                                           n_goals = goal_length,
                                           action_bounds = action_bounds,
                                           capacity = memory_size,
                                           action_size = action_length,
                                           batch_size = batch_size,
                                           actor_lr=actor_lr,
                                           critic_lr=critic_lr,
                                           gamma=gamma,
                                           tau=tau,
                                           k_future=k_future,
                                           file_name = "../networks/"+folder_name+"/network_pick_n_reach_goal_" + str(block_no) + ".pth")

                if (os.path.exists(block_wise_expansion[1].filename)):
                    block_wise_expansion[1].load_weights()
                    print("Picking up exiting network : ", block_wise_expansion[1].filename)

                transition = {}
                transition["next_state_info"] = copy.deepcopy(obs)
                transition["level_0_action"] = block_no
                transition["level_1_action"] = 2

                modified_transition = generate_desired_and_achieved_goal(transition)
                state_length = len(modified_transition["next_state_info"]["observation"])
                goal_length = len(modified_transition["desired_goal"])

                block_wise_expansion[2] = agent.Agent(n_states = state_length,
                                           n_actions = action_length,
                                           n_goals = goal_length,
                                           action_bounds = action_bounds,
                                           capacity = memory_size,
                                           action_size = action_length,
                                           batch_size = batch_size,
                                           actor_lr=actor_lr,
                                           critic_lr=critic_lr,
                                           gamma=gamma,
                                           tau=tau,
                                           k_future=k_future,
                                           file_name = "../networks/"+folder_name+"/network_release_and_lift_" + str(block_no)+ ".pth")
                if (os.path.exists(block_wise_expansion[2].filename)):
                    block_wise_expansion[2].load_weights()
                    print("Picking up exiting network : ", block_wise_expansion[2].filename)


            level_wise_mapping[block_no] = block_wise_expansion


        level_wise_mapping["action_space"] = "continuous"
        level_wise_mapping["action_length"] = action_length

        policy_program[level] = level_wise_mapping

    env.close()

    return policy_program

def generate_traces(traces_count = 10, gamma = 0.2):
    traces = {}

    env = gym.make("FetchStack4Stage3-v1")

    order_of_blocks = ['3', '2', '1','0']


    for trace_index in range(traces_count):
        obs = env.reset()
        option_order = order_of_blocks.copy()
        current_trace = {}
        execution_stack = []
        current_option = 0
        done = False
        option_time_count = 0
        t = 0
        while not done:
            # print("Execution stack size - ", len(execution_stack))
            # env.render()
            if not any(execution_stack):
                if any(option_order):
                    option_number = option_order.pop()

                    current_option = action_policies.place_object("object_" + str(option_number))
                    if current_option.initial_predicate(obs):
                        execution_stack.append(current_option)
                else:
                    current_option = action_policies.do_nothing()
                    execution_stack.append(current_option)
            else :
                option_time_count += 1
                option_action = execution_stack[-1].compute_option_policy(obs)

                sars = {}
                sars['state'] = copy.deepcopy(obs)
                sars['action'] = copy.deepcopy(option_action)

                obs, reward, done, info = env.step(option_action)
                reward = (reward + 4)/4
                sars['reward'] = reward
                current_trace[t] = sars
                if not any(execution_stack):
                    execution_stack.append

                if(execution_stack[-1].termination_condition(obs)) or (option_time_count > 60):
                    option_time_count = 0
                    # print("Popping execution stack")
                    execution_stack.pop()
            t += 1

        print("End reward - ", reward)

        traces[trace_index] = current_trace
    env.close()
    return traces

def generate_traces_multilevel(block_count = 3, traces_count = 10):

    h_policy = get_multilevel_policy(block_count, 1, folder_name="gen_demos")

    traces = {}
    env = gym.make("FetchStack" + str(block_count) + "Stage3-v1")
    order_of_blocks = []
    for i in range(block_count-1, -1, -1):
        order_of_blocks.append(str(i))
    order_low_level = [2, 1, 0]

    total = 0.0
    for trace_index in range(traces_count):
        obs = env.reset()

        option_order = copy.deepcopy(order_of_blocks)
        low_level_actions = []

        current_trace = {}
        execution_stack = []
        current_option = 0
        done = False
        option_number = -1
        option_time_count = 0
        t = 0
        while not done:

            # print("Execution Stack size - ", len(execution_stack))

            # env.render()
            if len(execution_stack) == 0:

                if any(option_order):
                    option_number = option_order.pop()

                    current_option = h_policy[len(execution_stack)]["uber_action"][int(option_number)]
                    low_level_actions = copy.deepcopy(order_low_level)

                    if current_option["pre"](obs):
                        execution_stack.append(current_option)
                        # print("Pushing tasks for object - ", option_number)

                else:
                    current_option = h_policy[len(execution_stack)]["uber_action"][block_count]
                    low_level_actions = []
                    execution_stack.append(current_option)
                    lower_option = h_policy[len(execution_stack)]["uber_action"][block_count]["uber_action"][len(order_low_level)]
                    execution_stack.append(lower_option)

                continue
            elif len(execution_stack) == 1:

                if any(low_level_actions):
                    action_number = low_level_actions.pop()

                    current_option = h_policy[len(execution_stack)]["uber_action"][int(option_number)]["uber_action"][action_number]
                    if current_option["pre"](obs):
                        execution_stack.append(current_option)
                        # print("Pushing sub action number - ", action_number)
                else:
                    low_level_actions = []
                    execution_stack.pop()
            else :
                option_time_count += 1
                option_action = execution_stack[-1]["action"](obs)

                sars = {}
                sars['state'] = copy.deepcopy(obs)
                sars['action'] = copy.deepcopy(option_action)

                obs, reward, done, info = env.step(option_action)
                reward = (reward + float(block_count))/float(block_count)
                reward = np.round(reward, 3)
                sars['reward'] = reward
                current_trace[t] = sars

                if not any(execution_stack):
                    execution_stack.append(current_option)

                if(execution_stack[-1]["post"](obs)) or (option_time_count > 80):
                    option_time_count = 0
                    execution_stack.pop()
            t += 1

        print("End reward - ", reward)
        total += reward

        traces[trace_index] = current_trace

    print("Average - ", total / float(len(traces)))
    env.close()
    return traces

def generate_desired_and_achieved_goal(transition, block_count = 3):

    block_no = transition["level_0_action"]
    height = 0.65

    if block_no != block_count :
        if transition["level_1_action"] == 0 :
            # Desired goal is the current location of the block, and open gripper
            # Achieved goal is the current location of the grip position, and gripper state

            desired_goal = np.concatenate([transition["next_state_info"]["annotated_obs"]["object_" + str(block_no)]["pos"],\
             np.asarray([1,1])])

            achieved_goal = np.concatenate([transition["next_state_info"]["annotated_obs"]["grip_pos"], \
            transition["next_state_info"]["annotated_obs"]["gripper_state"]])

            transition["desired_goal"] = desired_goal
            transition["achieved_goal"] = achieved_goal
        elif transition["level_1_action"] == 1 :
            # Desired goal is the current goal of the block we are interested in, and gripper state
            # Achieved Goal is the current gripper position, and gripper state

            ## MAY BE MAKE THIS CLOSED GRIP ????

            goal_states = action_policies.find_goal_states(transition["next_state_info"])

            desired_goal = np.concatenate([goal_states["object_"+str(block_no)] , \
            transition["next_state_info"]["annotated_obs"]["gripper_state"]])


            achieved_goal = np.concatenate( [transition["next_state_info"]["annotated_obs"]["object_"+str(block_no)]["pos"]\
             , transition["next_state_info"]["annotated_obs"]["gripper_state"]])

            transition["desired_goal"] = desired_goal
            transition["achieved_goal"] = achieved_goal

        elif transition["level_1_action"] == 2:

            # Desired goal is the grip position directly above the block goal position
            # Achieved goal is the current grip position

            desired_goal = transition["next_state_info"]["annotated_obs"]["object_" + str(block_no)]["pos"]
            desired_goal[2]  = height
            achieved_goal = transition["next_state_info"]["annotated_obs"]["grip_pos"]


            transition["desired_goal"] = desired_goal
            transition["achieved_goal"] = achieved_goal

    else:

        transition["desired_goal"] = transition["next_state_info"]["desired_goal"]
        transition["achieved_goal"] = transition["next_state_info"]["achieved_goal"]


    return transition

def compute_custom_reward(achieved_goal, goal):
    if np.linalg.norm(achieved_goal - goal) <= 0.04 :
        return np.array([0.0])
    else:
        return np.array([-1.0])

def get_flat_policy(block_count = 3, folder_name = "block_3_flat_level_1"):

    env = gym.make("FetchStack" + str(block_count) + "Stage3-v1")
    obs = env.reset()
    state_length = len(obs['observation']) + len(obs['achieved_goal'])

    action_length = block_count * 3 + 1 # 3 because of the number of pieces at level 1, and 1 because of the 'do nothing' action

    fetch_policy = policies.Policy(state_length, action_length, \
    actor_file = "../networks/"+folder_name+"/actor_network_flat.pkl",
    critic_file = "../networks/"+folder_name+"/critic_network_flat.pkl")


    policy_collection = {}
    policy_collection["policy"] = fetch_policy

    action_index = 0

    for block in range(block_count):

        option_0_details = {}
        option_0 = action_policies.reach_object("object_" + str(block))
        option_0_details["pre"] = option_0.predicate
        option_0_details["post"] = option_0.termination
        option_0_details["action"] = option_0.compute_option_policy
        policy_collection["option_" + str(action_index)] = option_0_details
        action_index += 1


        option_1_details = {}
        option_1 = action_policies.pick_and_reach_goal("object_" + str(block))
        option_1_details["pre"] = option_1.predicate
        option_1_details["post"] = option_1.termination
        option_1_details["action"] = option_1.compute_option_policy_2
        policy_collection["option_" + str(action_index)] = option_1_details
        action_index += 1


        option_2_details = {}
        option_2 = action_policies.release_and_lift("object_" + str(block))
        option_2_details["pre"] = option_2.predicate
        option_2_details["post"] = option_2.termination
        option_2_details["action"] = option_2.compute_option_policy
        policy_collection["option_" + str(action_index)] = option_2_details
        action_index += 1


    option_3_details = {}
    option_3 = action_policies.do_nothing()
    option_3_details["pre"] = option_3.initial_predicate
    option_3_details["post"] = option_3.termination_condition
    option_3_details["action"] = option_3.compute_option_policy
    policy_collection["option_" + str(action_index)] = option_3_details


    env.close()

    return policy_collection
