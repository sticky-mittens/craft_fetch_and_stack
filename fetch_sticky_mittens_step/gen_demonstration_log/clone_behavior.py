import sys
sys.path.append("../")
import copy
import torch
import numpy as np
import torch.optim as optim
import torch.nn as nn
from uber_policies import action_policies
import random

no_of_uber_actions = 4
def train_network_for_traces(network_file, traces, uber_actions):

    # Initialize network training stuff
    network = torch.load(network_file)
    learning_rate = 1e-4
    optimizer = optim.Adam(network.parameters(), lr = learning_rate)
    epoch_count = 100
    # criterion = nn.CrossEntropyLoss()
    criterion = nn.SmoothL1Loss()

    training_data = {}
    for trace_index in traces.keys():
        training_trace = produce_training_pairs(traces[trace_index], uber_actions)
        training_data[trace_index] = training_trace


    for epoch_index in range(epoch_count):

        for trace_id in training_data.keys():
            training_trace = training_data[trace_id]
            random.shuffle(training_trace)

            optimizer.zero_grad()
            loss_list = []

            for each_pair in training_trace:
                state_val = torch.tensor(each_pair[0]['observation'].astype(np.float32), requires_grad = False)
                goal_val = torch.tensor(each_pair[0]['desired_goal'].astype(np.float32), requires_grad = False)
                input = torch.cat((state_val, goal_val))


                action = each_pair[1]
                value = each_pair[2]

                # print("Action picked - ", action)
                # target = torch.zeros(len(uber_actions), dtype=torch.long, requires_grad = False)
                # target[action] = 1
                np_target = np.zeros(len(uber_actions))
                np_target[action] = value
                np_target = np_target.astype(np.float32)
                target = torch.from_numpy(np_target)


                prediction_ = network(input)
                # prediction = prediction_.unsqueeze(0)
                prediction = prediction_

                # val, index = prediction.max(1)

                loss = criterion(prediction, target)
                # print("Prediction - ", prediction, " target - ", target, "loss - ", loss)
                loss_list.append(loss)
                # sys.exit()
            # sys.exit()
            if any(training_trace):
                total_loss = torch.stack(loss_list, dim = 0).sum()
                print("At epoch - ", epoch_index ," loss - ", total_loss)
                total_loss.backward()
                optimizer.step()
        # sys.exit()

        torch.save(network, network_file)



def produce_training_pairs(trace, uber_actions, gamma = 0.2):
    # print("Trace received - ", trace.keys())

    track_index_reward = {}
    index = 0
    target_list = []
    trace_slice = trace.copy()

    current_time = 1
    while current_time in trace.keys():
        current_state = trace_slice[current_time]['state']
        score = False
        for each_action in uber_actions.keys():

            next_time, score = assign_score_to_decision(trace_slice, current_time, each_action)
            # print("At time - ", current_time, " for action - ", each_action, " score - ", score, " next time stamp - ", next_time, \
            # " reward - ", trace_slice[current_time]['reward'])

            if score:
                current_time = next_time
                if each_action != (no_of_uber_actions) :
                    index = len(target_list)
                    track_index_reward[index] = trace_slice[current_time]['reward']
                    target_list.append([current_state, each_action, 0])
                break

        if not score:
            current_time += 1
            while current_time not in trace.keys() and current_time < 2 * len(trace.keys()) :
                current_time += 1


    value = 0
    for index in reversed(sorted(track_index_reward.keys())):
        if value == 0:
            value = track_index_reward[index]
        else:
            value = track_index_reward[index] + gamma * value
        target_list[index][2] = value

    # print("Target values : ")
    # for each_target in target_list:
    #     print(each_target[1], " ", each_target[2])
    # sys.exit()

    return target_list

def assign_score_to_decision(trace, time_index, uber_action):

    current_state = trace[time_index]['state']
    uber_transition = estimate_uber_output(current_state, uber_action)
    time_stamp, score = find_if_state_in_trace(uber_transition, trace, time_index+1, uber_action)

    return time_stamp, score

def find_if_state_in_trace(template_state, trace, starting_time, uber_action, thresh = 0.05):

    # Basically checks if any of the block positions match
    goals = action_policies.find_goal_states(template_state)
    init_unsat = False

    if starting_time not in trace.keys():
        return None, False

    if uber_action in range(0,no_of_uber_actions):
        block_position = trace[starting_time]['state']['annotated_obs']['object_' + str(uber_action)]['pos']
        rel_pos = goals["object_" + str(uber_action)] - block_position
        dist = action_policies.find_dist(rel_pos)
        if dist > thresh :
            init_unsat = True

    for each_time_stamp in trace.keys():
        if each_time_stamp >= starting_time :
            if uber_action == no_of_uber_actions :
                current_grip_pos = trace[each_time_stamp]['state']['annotated_obs']['grip_pos']
                template_grip_pos = template_state['annotated_obs']['grip_pos']
                rel_pos = current_grip_pos - template_grip_pos
                grip_diff = action_policies.find_dist(rel_pos)
                if grip_diff < thresh :
                    return each_time_stamp, True
            elif init_unsat:
                current_state = trace[each_time_stamp]
                block_position = current_state['state']['annotated_obs']['object_' + str(uber_action)]['pos']
                goal = goals["object_"+str(uber_action)]
                rel_pos = goal - block_position
                dist = action_policies.find_dist(rel_pos)
                if dist < thresh :
                    return each_time_stamp, True

    return None, False


def estimate_uber_output(current_state, uber_action):
    if uber_action == no_of_uber_actions :
        return_state = copy.deepcopy(current_state)
        return return_state

    return_state = copy.deepcopy(current_state)
    object_name = "object_" + str(uber_action)
    object_of_interest = return_state["annotated_obs"][object_name]

    # Find the current goals for each block
    object_goals = action_policies.find_goal_states(return_state)

    # Check if the block is in destination
    new_position = object_goals[object_name]
    return_state["annotated_obs"][object_name]["pos"] = new_position

    return return_state
