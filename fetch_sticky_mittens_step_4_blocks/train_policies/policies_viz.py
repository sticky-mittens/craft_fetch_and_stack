import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Bernoulli
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
import random
import math
import sys
import os
import time
import copy

sys.path.append("../")
from gen_demonstration_log import perform_task
from gen_demonstration_log import clone_behavior
from uber_policies import action_policies
import os.path
from os import path
import pickle


# Some class which defines a model
# Takes in some training samples and updates the model slowly( target, network)
# Can execute the model on sample inputs .
# Also save it to the disk and pull it back up

# Source - motivation from  : https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html

def compute_return(trajectories, gamma = 0.99):
    # Assumption, the trajectories are a collection of lists
    # Each list, is in turn a list of tuples : (state, action, reward)

    trajectory_values = {}
    for trajectory_index in trajectories.keys():
        trajectory = trajectories[trajectory_index]
        no_of_steps = len(trajectory)

        trace_value = {}
        return_val = 0
        for step_index in reversed(range(no_of_steps)):
            reward = trajectory[step_index]['reward']
            return_val = reward + gamma * return_val
            trace_value[step_index] = return_val

        trajectory_values[trajectory_index] = trace_value
    return trajectory_values

class policyNet(nn.Module):
    def __init__(self, no_of_inputs = 10, no_of_outputs = 5):
        self.no_of_layers = 2
        self.no_of_hidden_units = 100 # prev, 100
        super(policyNet, self).__init__()

        self.lin_trans = []

        self.lin_trans.append(nn.Linear(no_of_inputs, self.no_of_hidden_units))
        self.lin_trans.append(torch.nn.ReLU())
        for layer_index in range(self.no_of_layers-2):
            self.lin_trans.append(nn.Linear(self.no_of_hidden_units, self.no_of_hidden_units))
            self.lin_trans.append(torch.nn.ReLU())

        self.lin_trans.append(nn.Linear(self.no_of_hidden_units, no_of_outputs))
        self.lin_trans.append(torch.nn.Sigmoid())
        self.lin_trans.append(torch.nn.Softmax())

        self.model = torch.nn.Sequential(*self.lin_trans)

    def forward(self, x):
        y = self.model(x)
        return y

class criticNet(nn.Module):
    def __init__(self, no_of_inputs = 10, no_of_outputs = 5):
        self.no_of_layers = 4   # For 4 blocks it's probably better to have 5 layers. This is from the paper
        self.no_of_hidden_units = 300 # prev, 300

        super(criticNet, self).__init__()

        self.lin_trans = []

        self.lin_trans.append(nn.Linear(no_of_inputs, self.no_of_hidden_units))
        self.lin_trans.append(torch.nn.ReLU())
        for layer_index in range(self.no_of_layers-2):
            self.lin_trans.append(nn.Linear(self.no_of_hidden_units, self.no_of_hidden_units))
            self.lin_trans.append(torch.nn.ReLU())
        self.lin_trans.append(nn.Linear(self.no_of_hidden_units, no_of_outputs))
        # self.lin_trans.append(torch.nn.ReLU())
        # self.lin_trans.append(torch.nn.Softmax())
        # self.lin_trans.append(torch.nn.Sigmoid())

        self.model = torch.nn.Sequential(*self.lin_trans)

    def forward(self, x):
        y = self.model(x)
        return y


class ContPolicyNet(nn.Module):
    def __init__(self, no_of_inputs = 10, no_of_outputs = 5):
        self.no_of_layers = 4
        self.no_of_hidden_units = 256
        super(ContPolicyNet, self).__init__()

        self.lin_trans = []

        self.lin_trans.append(nn.Linear(no_of_inputs, self.no_of_hidden_units))
        self.lin_trans.append(torch.nn.ReLU())
        for layer_index in range(self.no_of_layers-2):
            self.lin_trans.append(nn.Linear(self.no_of_hidden_units, self.no_of_hidden_units))
            self.lin_trans.append(torch.nn.ReLU())

        self.lin_trans.append(nn.Linear(self.no_of_hidden_units, no_of_outputs))
        self.lin_trans.append(torch.nn.Tanh())


        self.model = torch.nn.Sequential(*self.lin_trans)

    def forward(self, x):
        y = self.model(x)
        # y = torch.mul(y, 50)
        return y


class ContCriticNet(nn.Module):
    def __init__(self, no_of_inputs = 10, no_of_outputs = 5):
        self.no_of_layers = 4
        self.no_of_hidden_units = 256

        super(ContCriticNet, self).__init__()

        self.lin_trans = []

        self.lin_trans.append(nn.Linear(no_of_inputs, self.no_of_hidden_units))
        self.lin_trans.append(torch.nn.ReLU())
        for layer_index in range(self.no_of_layers-2):
            self.lin_trans.append(nn.Linear(self.no_of_hidden_units, self.no_of_hidden_units))
            self.lin_trans.append(torch.nn.ReLU())
        self.lin_trans.append(nn.Linear(self.no_of_hidden_units, no_of_outputs))
        # self.lin_trans.append(torch.nn.ReLU())
        # self.lin_trans.append(torch.nn.Softmax())
        # self.lin_trans.append(torch.nn.Sigmoid())

        self.model = torch.nn.Sequential(*self.lin_trans)

    def forward(self, x):
        y = self.model(x)
        return y

class ContZeroPolicyNet(nn.Module):
    def __init__(self, no_of_inputs = 10, no_of_outputs = 5):
        super(ContZeroPolicyNet, self).__init__()
        self.output = torch.zeros(no_of_outputs, requires_grad = False)

    def forward(self, x):
        y = self.output
        return y

class ContZeroCriticNet(nn.Module):
    def __init__(self, no_of_inputs = 10, no_of_outputs = 5):
        super(ContZeroCriticNet, self).__init__()
        self.output = torch.zeros(no_of_outputs, requires_grad = False)

    def forward(self, x):
        y = self.output
        return y


class ContZeroPolicy():
    def __init__(self, input_count, action_count):
        self.actor_net = ContZeroPolicyNet(input_count, action_count)
        self.critic_net = ContZeroCriticNet(input_count + action_count, 1)

    def save_to_disk(self):
        pass

class Policy():
    def __init__(self, input_count, action_count, actor_file, critic_file):
        self.actor_file = actor_file
        self.critic_file = critic_file
        # EDIT
        self.actor_learning_rate = 1e-4
        self.critic_learning_rate = 1e-4
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.traces = {}

        self.demos_file = "../gen_demonstration_log/demos.p"


        if (os.path.exists(self.actor_file) and os.path.exists(self.critic_file)):
            self.actor_net = torch.load(self.actor_file, map_location = self.device)
            self.critic_net = torch.load(self.critic_file, map_location = self.device)
            # print("Picked the networks from previous iterations.. ")
        else :
            self.actor_net = policyNet(input_count, action_count)
            self.actor_target_net = policyNet(input_count, action_count)

            # self.critic_net = criticNet(input_count + action_count, 1)
            self.critic_net = criticNet(input_count, action_count)
            self.critic_target_net = criticNet(input_count, action_count)

            self.actor_net = self.actor_net.to(self.device)
            self.critic_net = self.critic_net.to(self.device)
            self.actor_target_net = self.actor_target_net.to(self.device)
            self.critic_target_net = self.critic_target_net.to(self.device)

            print("Built new networks")

        self.optimizer_actor = optim.Adam(self.actor_net.parameters(), lr = self.actor_learning_rate)
        self.optimizer_critic = optim.Adam(self.critic_net.parameters(), lr = self.critic_learning_rate)

        if path.exists(self.demos_file):
            with open(self.demos_file, 'rb') as fp:
                self.traces = pickle.load(fp)

        self.training_data = {}

    def update_policy(self, exp_collection, gamma = 0.2):

        if not exp_collection:
            return

        criterion = nn.SmoothL1Loss()

        batch_size = math.ceil(0.2 * len(exp_collection)) # 0.1
        # print("Batch size - ", batch_size)

        # Build a batch of traning data
        actor_loss_list = []
        critic_loss_list = []

        self.optimizer_actor.zero_grad()
        self.optimizer_critic.zero_grad()

        for index in range(batch_size):

            trans = random.choice(list(exp_collection.values()))
            if trans['terminal'] :
                target_val = trans['reward']
            else:
                next_val, next_action = self.critic_net.forward(trans['next_state_goal']).max(0)
                target_val = trans['reward'] + gamma * next_val.clone().detach()

            prediction = self.critic_net.forward(trans['current_state_goal'])
            current_val, current_action = prediction.max(0)

            np_target = np.zeros(len(self.uber_actions))
            np_target[trans['action']] = target_val
            np_target = np_target.astype(np.float32)
            torch_target = torch.from_numpy(np_target).to(self.device)

            # print("For action - ", trans['action'], " value - ", target_val)


            # print("current critic val - ", prediction, " target - ", torch_target)
            loss = criterion(prediction, torch_target).to(self.device)

            # action_log = trans['action_log']
            # actor_loss_list.append(action_log * adv.clone().detach())
            critic_loss_list.append(loss)

        # Take a gradient step

        # actor_loss = torch.stack(actor_loss_list, dim = 0).sum()
        critic_loss = torch.stack(critic_loss_list, dim = 0).sum()

        critic_loss.backward()
        self.optimizer_critic.step()

        self.optimizer_critic.zero_grad()
        loss_list = []
        count = 50
        entry_list = list(self.training_data.keys())
        # for trace_id in self.training_data.keys():
        for index in range(count):
            trace_id = random.choice(entry_list)
            training_trace = self.training_data[trace_id]
            random.shuffle(training_trace)

            for each_pair in training_trace:
                state_val = torch.tensor(each_pair[0]['observation'].astype(np.float32), requires_grad = False)
                goal_val = torch.tensor(each_pair[0]['desired_goal'].astype(np.float32), requires_grad = False)
                input = torch.cat((state_val, goal_val)).to(self.device)


                action = each_pair[1]
                value = each_pair[2]

                np_target = np.zeros(len(self.uber_actions))
                np_target[action] = value
                np_target = np_target.astype(np.float32)
                target = torch.from_numpy(np_target).to(self.device)


                prediction_ = self.critic_net.forward(input)
                prediction = prediction_

                loss = criterion(prediction, target)
                # print("Prediction - ", prediction, " target - ", target, "loss - ", loss)
                loss_list.append(loss)
                # sys.exit()

        total_loss = torch.stack(loss_list, dim = 0).sum()
        total_loss.backward()
        self.optimizer_critic.step()

    def save_to_disk(self):
        torch.save(self.actor_net, self.actor_file)
        torch.save(self.critic_net, self.critic_file)




class Continuous_Policy():
    def __init__(self, input_count, action_count, actor_file, critic_file):
        self.actor_file = actor_file
        self.critic_file = critic_file
        # EDIT
        self.tau = 1e-2
        self.actor_learning_rate = 1e-3
        self.critic_learning_rate = 1e-3
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.traces = {}

        self.demos_file = "../gen_demonstration_log/demos.p"


        if (os.path.exists(self.actor_file) and os.path.exists(self.critic_file)):
            self.actor_net = torch.load(self.actor_file, map_location = self.device)
            self.critic_net = torch.load(self.critic_file, map_location = self.device)
            # print("Picked the networks from previous iterations.. ")

            self.target_actor_net = ContPolicyNet(input_count, action_count)
            self.target_critic_net = ContCriticNet(input_count + action_count, 1)

            self.target_actor_net = self.target_actor_net.to(self.device)
            self.target_critic_net = self.target_critic_net.to(self.device)

            hard_update(self.target_actor_net, self.actor_net)
            hard_update(self.target_critic_net, self.critic_net)

        else :
            self.actor_net = ContPolicyNet(input_count, action_count)
            self.target_actor_net = ContPolicyNet(input_count, action_count)

            # self.critic_net = criticNet(input_count + action_count, 1)
            self.critic_net = ContCriticNet(input_count + action_count, 1)
            self.target_critic_net = ContCriticNet(input_count + action_count, 1)

            hard_update(self.target_actor_net, self.actor_net)
            hard_update(self.target_critic_net, self.critic_net)

            self.actor_net = self.actor_net.to(self.device)
            self.critic_net = self.critic_net.to(self.device)
            self.target_actor_net = self.target_actor_net.to(self.device)
            self.target_critic_net = self.target_critic_net.to(self.device)

            print("Built new networks")

        self.optimizer_actor = optim.Adam(self.actor_net.parameters(), lr = self.actor_learning_rate)
        self.optimizer_critic = optim.Adam(self.critic_net.parameters(), lr = self.critic_learning_rate)

        if path.exists(self.demos_file):
            with open(self.demos_file, 'rb') as fp:
                self.traces = pickle.load(fp)

        self.training_data = {}

    def soft_update(self):
        soft_update(self.target_actor_net, self.actor_net, self.tau)
        soft_update(self.target_critic_net, self.critic_net, self.tau)

    def hard_update(self):
        hard_update(self.target_actor_net, self.actor_net)
        hard_update(self.target_critic_net, self.critic_net)

    def save_to_disk(self):
        torch.save(self.actor_net, self.actor_file)
        torch.save(self.critic_net, self.critic_file)


class Hierarchical_Policies():
    ''' 1. Defines multiple policies
        2. Get the pre and post conditions at multiple levels
        3. Given a trajectory breaks it up into multiple stages depending on the policy level.
    '''


    def __init__(self, no_of_blocks, levels_count, folder_name, demos_file = "../gen_demonstration_log/demos.p"):
        self.levels = levels_count
        self.block_count = no_of_blocks
        self.lower_actions = 3
        self.decay_rate = 10000 # increase to reduce exploration and only exploit

        self.multilevel_policy = perform_task.get_multilevel_policy(no_of_blocks, self.levels-1, folder_name)

        self.execution_stack = []
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.traces = {}

        self.demos_file = demos_file
        if path.exists(self.demos_file):
            with open(self.demos_file, 'rb') as fp:
                self.traces = pickle.load(fp)


        self.book_keeping_reset()

        self.gamma = {}
        self.gamma[0] = 0.2
        self.gamma[1] = 0.8
        self.gamma[2] = 0.99

        self.last_count = {0:0, 1:{0:0, 1:0, 2:0, 3:0, 4:0}}

        self.network_wise_episode_count = {}
        self.network_wise_episode_count[0] = 0

        self.network_wise_episode_count[1] = {}

        self.network_wise_episode_count[1][0] = 0
        self.network_wise_episode_count[1][1] = 0
        self.network_wise_episode_count[1][2] = 0
        self.network_wise_episode_count[1][3] = 0
        self.network_wise_episode_count[1][4] = 0

        lower_counts = {}
        for each_block in range(self.block_count) :
            block_wise = {}
            for each_action in range(self.lower_actions) :
                block_wise[each_action] = 0
            lower_counts[each_block] = block_wise

        self.network_wise_episode_count[2] = lower_counts
        self.last_count[2] = copy.deepcopy(lower_counts)

        # Loading up the demonstrations
        self.training_data = {}
        self.timing_data = {}
        # Take care of this part
        if any(self.traces):
            for trace_index in self.traces.keys():
                if self.filter_trace(self.traces[trace_index]):
                    training_trace, training_time = self.view_trajectory_at_multiple_level(self.traces[trace_index], 1)
                    # training_trace = clone_behavior.produce_training_pairs(self.traces[trace_index], self.uber_actions)
                    self.training_data[trace_index] = training_trace
                    self.timing_data[trace_index] = training_time

        print("Number of slices of demonstrations - ", len(self.training_data))



    def filter_trace(self, trace):
        last_post_cond = self.multilevel_policy[0]["uber_action"][self.block_count-1]["post"]
        for each_time in trace.keys():
            # if trace[each_time]["reward"] > 0.99 :
            if last_post_cond(trace[each_time]["state"]):
                return True

        return False

    def book_keeping_reset(self):
        self.execution_stack = []
        self.option_timeout = {}
        self.option_timeout[0] = 0
        self.option_timeout[1] = 0
        self.top_action = None
        self.state_tracking = {}
        self.actual_action = np.zeros(self.multilevel_policy[2]["action_length"])


    def update_policy(self, exp_collection, level = 0):

        if not exp_collection:
         return

        if level == 2:
            self.update_continuous_policy(exp_collection)
            return

        criterion = nn.SmoothL1Loss()

        if level == 0 :
            batch_size = math.ceil(0.2 * len(exp_collection)) # 0:0.5
        elif level == 1:
            batch_size = math.ceil(0.2 * len(exp_collection))

        # Build a batch of traning data
        # critic_loss_list = []

        # self.optimizer_actor.zero_grad()
        current_policy = None
        current_action_length = None

        critic_loss_list = {}
        actor_loss_list = {}


        # Compute the loss for each policy
        transition_collection = []

        for index in range(batch_size):

            trans = random.choice(list(exp_collection.values()))

            if level == 0 :
                top_action = 0
                current_policy = self.multilevel_policy[level]["policy"]
                current_action_length = self.multilevel_policy[level]["action_length"]
            elif level == 1:
                top_action = trans["top_action"]
                current_policy = self.multilevel_policy[level]["uber_action"][top_action]["policy"]
                current_action_length = self.multilevel_policy[level]["uber_action"][top_action]["action_length"]
            else:
                print("Level not implemented yet.")
                sys.exit()


            transition_collection.append(trans)
            if trans['terminal'] :
                target_val = trans['reward']
            else:
                next_val, next_action = current_policy.critic_net.forward(trans['next_state_goal']).max(0)
                target_val = trans['reward'] + self.gamma[level] * next_val.clone().detach()


            prediction = current_policy.critic_net.forward(trans['current_state_goal'])
            current_val, current_action = prediction.max(0)

            np_target = np.zeros(current_action_length)
            np_target[trans['action']] = target_val
            np_target = np_target.astype(np.float32)
            torch_target = torch.from_numpy(np_target).to(self.device)

            # print(" train Target - ", torch_target, " block no - ", top_action)

            loss = criterion(prediction, torch_target).to(self.device)

            if top_action in critic_loss_list.keys():
                critic_loss_list[top_action].append(loss)
            else:
                critic_loss_list[top_action] = []
                critic_loss_list[top_action].append(loss)


        # Do the back propagation here
        for each_key in critic_loss_list.keys():
            if any(critic_loss_list[each_key]):
                if level == 0 :
                    current_policy = self.multilevel_policy[level]["policy"]
                    current_policy.optimizer_critic.zero_grad()
                    critic_loss = torch.stack(critic_loss_list[each_key], dim = 0).sum()
                    critic_loss.backward()
                    current_policy.optimizer_critic.step()

                elif level == 1:
                    current_policy = self.multilevel_policy[level]["uber_action"][each_key]["policy"]
                    current_policy.optimizer_critic.zero_grad()
                    critic_loss = torch.stack(critic_loss_list[each_key], dim = 0).sum()
                    critic_loss.backward()
                    current_policy.optimizer_critic.step()
                    # print("experience loss - ", critic_loss)

                else:
                    print("Level not implemented yet.")
                    sys.exit()

        # Training against the demonstrations here


        loss_list = []
        if level == 0:
            count = 50 # Level 0: 50,10  Level 1 : 100, 0
        elif level == 1:
            count = 10
        entry_list = list(self.training_data.keys())

        if level == 0:
            current_policy = self.multilevel_policy[level]["policy"]
            current_action_length = self.multilevel_policy[level]["action_length"]
            current_policy.optimizer_critic.zero_grad()

            for index in range(count):
                trace_id = random.choice(entry_list)
                training_trace = self.training_data[trace_id][level]

                for each_pair in training_trace:
                    state_val = torch.tensor(each_pair[0]['observation'].astype(np.float32), requires_grad = False)
                    goal_val = torch.tensor(each_pair[0]['desired_goal'].astype(np.float32), requires_grad = False)
                    input = torch.cat((state_val, goal_val)).to(self.device)


                    action = each_pair[1]
                    value = each_pair[2]

                    # print("Action picked - ", action)
                    # target = torch.zeros(len(uber_actions), dtype=torch.long, requires_grad = False)
                    # target[action] = 1
                    np_target = np.zeros(current_action_length)
                    np_target[action] = value
                    np_target = np_target.astype(np.float32)
                    target = torch.from_numpy(np_target).to(self.device)


                    prediction_ = current_policy.critic_net.forward(input)
                    # prediction = prediction_.unsqueeze(0)
                    prediction = prediction_

                    # val, index = prediction.max(1)

                    loss = criterion(prediction, target)
                    # print("Prediction - ", prediction, " target - ", target, "loss - ", loss)
                    loss_list.append(loss)
                    # sys.exit()
                # sys.exit()
                # if any(training_trace):
            # print("Loss list - ", loss_list)
            total_loss = torch.stack(loss_list, dim = 0).sum()
            # print("Demonstration loss - ", total_loss)
            total_loss.backward()
            current_policy.optimizer_critic.step()

        elif level == 1:
            for block_picked in range(self.block_count):
                current_policy = self.multilevel_policy[level]["uber_action"][block_picked]["policy"]
                current_action_length = self.multilevel_policy[level]["uber_action"][block_picked]["action_length"]
                current_policy.optimizer_critic.zero_grad()
                loss_list = []

                for index in range(count):
                    trace_id = random.choice(entry_list)
                    # print('[DEBUG] ', trace_id, level, block_picked, self.training_data.keys(), self.training_data[trace_id].keys())
                    # print('[DEBUG] ', self.training_data[trace_id][level].keys())
                    # print('[DEBUG] ', self.training_data[trace_id][level][block_picked])

                    if block_picked not in self.training_data[trace_id][level].keys():
                        continue
                    training_trace = self.training_data[trace_id][level][block_picked]

                    for each_pair in training_trace:
                        state_val = torch.tensor(each_pair[0]['observation'].astype(np.float32), requires_grad = False)
                        goal_val = torch.tensor(each_pair[0]['desired_goal'].astype(np.float32), requires_grad = False)
                        input = torch.cat((state_val, goal_val)).to(self.device)


                        action = each_pair[1]
                        value = each_pair[2]

                        np_target = np.zeros(current_action_length)
                        np_target[action] = value
                        np_target = np_target.astype(np.float32)
                        target = torch.from_numpy(np_target).to(self.device)

                        # print("Target - ", target, " block - ", block_picked)

                        prediction_ = current_policy.critic_net.forward(input)
                        prediction = prediction_


                        loss = criterion(prediction, target)
                        loss_list.append(loss)

                    # print("\n")
                if count > 0:
                    total_loss = torch.stack(loss_list, dim = 0).sum()
                    # print("Demonstration loss - ", total_loss)
                    total_loss.backward()
                    current_policy.optimizer_critic.step()
        else:
            print("Level not implemented yet")


    def compute_continuous_action(self, current_state, train_mode = False):

        # TRANFORM EDIT : Use agent's choose action thing here


        # exploration_number = np.exp(-1 * self.decay_rate * 1e-3 * \
        # (self.network_wise_episode_count[level][self.state_tracking["level_0_action"]][self.state_tracking["level_1_action"]] + 1)) # 0.2
        exploration_number = 0.1

        modified_state = perform_task.generate_desired_and_achieved_goal(current_state)

        current_policy = self.multilevel_policy[2][modified_state["level_0_action"]]\
        [modified_state["level_1_action"]]

        action = current_policy.choose_action(modified_state["next_state_info"]['observation'],\
        modified_state['desired_goal'], train_mode)


        return action, modified_state


    def update_continuous_policy(self, episodic_memory):

        updates_count = 40

        agent_wise_mb = {}

        reward_sum = 0
        count = 0
        for episode_id in episodic_memory.keys():

            episode = episodic_memory[episode_id]
            episode_slices_by_time = self.slice_actions_by_time(episode)

            for each_slice in episode_slices_by_time :

                episode_dict = {
                    "state": [],
                    "action": [],
                    "info": [],
                    "achieved_goal": [],
                    "desired_goal": [],
                    "next_state": [],
                    "next_achieved_goal": []}

                level_0_action = -1
                level_1_action = -1
                for time in each_slice.keys():
                    trans = each_slice[time]

                    level_0_action = trans["level_0_action"]
                    level_1_action = trans["level_1_action"]

                    episode_dict["state"].append(trans["current_state_info"]["observation"].copy())
                    episode_dict["action"].append(trans["action"].copy())
                    episode_dict["achieved_goal"].append(trans["achieved_goal"].copy())
                    episode_dict["desired_goal"].append(trans["desired_goal"].copy())

                episode_dict["next_state"] = episode_dict["state"][1:]
                episode_dict["next_achieved_goal"] = episode_dict["achieved_goal"][1:]

                reward_sum += perform_task.compute_custom_reward(trans["achieved_goal"], trans["desired_goal"])
                count += 1

                if len(episode_dict) > 1:
                    if level_0_action not in agent_wise_mb.keys():
                        agent_wise_mb[trans["level_0_action"]] = {}

                    if level_1_action not in agent_wise_mb[level_0_action] :
                        agent_wise_mb[level_0_action][level_1_action] = []

                    agent_wise_mb[level_0_action][level_1_action].append(copy.deepcopy(episode_dict))

        print("Average reward = ", reward_sum / float(count))
        for block in agent_wise_mb.keys():
            for action_id in agent_wise_mb[block].keys():
                current_policy = self.multilevel_policy[2][block][action_id]
                current_policy.store(agent_wise_mb[block][action_id])

                if len(current_policy.memory) > current_policy.batch_size:
                    for n_update in range(updates_count):
                        actor_loss, critic_loss = current_policy.train()
                    current_policy.update_networks()


    def expand_transitions(self, episodic_memory, batch_size = 100, HER_fraction = 0.3):

        ep_indices = np.random.randint(0, len(list(episodic_memory.keys())), batch_size)

        all_transitions = []

        orig_count = 0
        her_count = 0

        for episode_id in ep_indices:
            episode = episodic_memory[episode_id]
            episode_slices_by_time = self.slice_actions_by_time(episode)

            for each_slice in episode_slices_by_time :
                for current_time in each_slice.keys():
                    if np.random.uniform() < HER_fraction :
                        all_transitions.append(each_slice[current_time])
                        orig_count += 1
                        # Add individual transitions first

                        filtered_time = [time for time in each_slice.keys() if time > current_time]
                        if len(filtered_time) < 1 :
                            continue
                        future_times = random.sample(filtered_time, max(int(0.05 * len(filtered_time)),1) )
                        # future_times = random.sample(filtered_time, 1)

                        for future_time in future_times:
                            current_trans = copy.deepcopy(each_slice[current_time])
                            future_trans = copy.deepcopy(each_slice[future_time])
                            new_trans = self.make_HER_trans(current_trans, future_trans)

                            all_transitions.append(new_trans)
                            her_count += 1

        print("Orig goal transitions - ", orig_count, " HER transitions - ", her_count)
        return all_transitions

    def make_HER_trans(self, current_trans, future_trans):
        new_trans = {}

        new_trans = copy.deepcopy(current_trans)
        new_trans["desired_goal"] = future_trans["achieved_goal"]

        if np.linalg.norm(new_trans["achieved_goal"] - new_trans["desired_goal"]) < 0.05 :
            new_trans["reward"] = 0.0
            new_trans["terminal"] = True


        return new_trans

    def slice_actions_by_time(self, episode):
        # Bascically starting from the current time stamp, find till what time
        # the level 0 and level 1 action goes on till

        episode_slices = []
        time_list = list(episode.keys())
        time_list.sort()
        time = time_list[0]
        while time in episode.keys():
            level_0_action = episode[time]["level_0_action"]
            level_1_action = episode[time]["level_1_action"]
            current_slice = {}
            while time in episode.keys() and episode[time]["level_0_action"] == level_0_action \
            and episode[time]["level_1_action"] == level_1_action :
                current_slice[time] = episode[time]
                time += 1

            if any(current_slice):
                episode_slices.append(current_slice)

        return episode_slices


    def view_trajectory_at_multiple_level(self, trajectory, level = 1):

        trajectory_slices = {}
        trajectory_times = {}
        trace_slice = copy.deepcopy(trajectory)

        all_times = list(trajectory.keys())
        all_times.sort()
        time_range = [all_times[0], all_times[-1]]

        if level == 0:
            post_conditions = {}
            for each_key in self.multilevel_policy[level]["uber_action"].keys():
                post_conditions[each_key] = self.multilevel_policy[level]["uber_action"][each_key]["post"]

            trajectory_slices[level], trajectory_times[level] = self.slice_trajectory(trace_slice, time_range, \
            post_conditions, level)

        elif level == 1:

            trajectory_slices, trajectory_times = self.view_trajectory_at_multiple_level(trace_slice, level-1)
            trajectory_slices[level] = {}
            trajectory_times[level] = {}

            # For each slice further sub-divide it into more pieces.
            for each_block in list(trajectory_times[level-1].keys())[0:-1]:
                post_conditions = {}
                for each_key in self.multilevel_policy[level]["uber_action"][each_block]["uber_action"].keys():
                    post_conditions[each_key] = self.multilevel_policy[level]["uber_action"][each_block]["uber_action"][each_key]["post"]

                # for sub_level_key in trajectory_times[level-1][each_block].keys():
                time_duration = trajectory_times[level-1][each_block]

                trajectory_slices[level][each_block], trajectory_times[level][each_block] = \
                self.slice_trajectory(trace_slice, time_duration, post_conditions, level)


        elif level == 2:
            print("Level 3 not implemented yet")
            sys.exit()



        return trajectory_slices, trajectory_times


    def execute_policy(self, level, obs, episode_index):


        # print("Executing policy---- , length of execution stack - ", len(self.execution_stack))
        made_uber_transition = False
        current_state = obs['observation'].astype(np.float32)
        current_goal = obs['desired_goal'].astype(np.float32)
        achieved_goal = obs['achieved_goal'].astype(np.float32)
        torch_state = torch.cat((torch.tensor(current_state), torch.tensor(current_goal)))
        torch_state = torch_state.to(self.device)



        if len(self.execution_stack) == 0:

            exploration_number = np.exp(-1 * self.decay_rate * 1e-3 * \
            (self.network_wise_episode_count[0] + 1)) # 0.2


            if level == 0 and episode_index > self.last_count[level] :
                self.last_count[level] = episode_index
                self.network_wise_episode_count[level] += 1


            if np.random.uniform(0,1) <= exploration_number and level == 0:
                action = np.random.randint(self.block_count)
                python_action = action
            else:
                current_policy = self.multilevel_policy[len(self.execution_stack)]["policy"]
                torch_distribution = current_policy.critic_net.forward(torch_state)
                val, action = torch_distribution.max(0)
                # print("Action - ", action)
                python_action = action.item()

            # Note this assignment is only for the pre post condition checking not for execution
            current_option = self.multilevel_policy[len(self.execution_stack)]["uber_action"][python_action]
            current_option["id"] = python_action

            # print("Action chosen at level 0 : ", python_action)

            if current_option["pre"](obs):
                self.execution_stack.append(current_option)
            else:
                # print("Choosing null because pre condition unsat")
                null_option = self.multilevel_policy[len(self.execution_stack)]["uber_action"][self.block_count]
                null_option["id"] = self.block_count
                self.execution_stack.append(null_option)

                lower_option = self.multilevel_policy[len(self.execution_stack)]["uber_action"][self.block_count]["uber_action"][self.lower_actions]
                self.execution_stack.append(lower_option)

            # Track transitions for level 0
            if level == 0:
                if not any(self.state_tracking):
                    self.state_tracking["top_action"] = None
                    self.state_tracking["current_state_goal"] = torch_state
                    self.state_tracking["action"] = python_action

        elif len(self.execution_stack) == 1 :

            if level == 0:
                self.option_timeout[0] += 1
                self.actual_action = self.execution_stack[-1]["action"](obs)
                if(self.execution_stack[-1]["post"](obs)) or (self.option_timeout[0] > 80):

                    if any(self.state_tracking) :
                        made_uber_transition = True
                        current_goal = obs['desired_goal'].astype(np.float32)
                        self.state_tracking["next_state_goal"] = torch_state
                        self.state_tracking["achieved_goal"] = current_goal

                    self.option_timeout[0] = 0
                    self.execution_stack.pop()

            elif level == 1:

                level_0_action = self.execution_stack[-1]["id"]

                exploration_number = np.exp(-1 * self.decay_rate * 1e-3 * \
                (self.network_wise_episode_count[1][level_0_action] + 1)) # 0.2


                if episode_index > self.last_count[1][level_0_action] :
                    self.last_count[1][level_0_action] = episode_index
                    self.network_wise_episode_count[1][level_0_action] += 1

                if(self.execution_stack[-1]["post"](obs)) or (self.option_timeout[0] > 80):

                    self.option_timeout[0] = 0
                    self.execution_stack.pop()
                    left, right = self.execute_policy(level, obs, episode_index)

                    return left, right

                if np.random.uniform(0,1) <= exploration_number and level == 1:
                    action = np.random.randint(self.multilevel_policy[len(self.execution_stack)]["uber_action"][level_0_action]["action_length"])
                    python_action = action
                else:
                    current_policy = self.multilevel_policy[len(self.execution_stack)]["uber_action"][level_0_action]["policy"]
                    torch_distribution = current_policy.critic_net.forward(torch_state)
                    val, action = torch_distribution.max(0)
                    # print("Action - ", action)
                    python_action = action.item()


                # print("\t \t Action chosen at level 1 - ", python_action)

                # Note this assignment is only for the pre post condition checking not for execution
                # The action field can be used as of now. In the absence of no defined low level
                current_option = self.multilevel_policy[len(self.execution_stack)]["uber_action"][level_0_action]["uber_action"][python_action]
                current_option["id"] = python_action

                if current_option["pre"](obs):
                    self.execution_stack.append(current_option)
                else:
                    lower_option = self.multilevel_policy[len(self.execution_stack)]["uber_action"][self.block_count]["uber_action"][self.lower_actions]
                    self.execution_stack.append(lower_option)

                # track transitions for level
                if not any(self.state_tracking):
                    self.state_tracking["top_action"] = level_0_action
                    self.state_tracking["current_state_goal"] = torch_state
                    self.state_tracking["action"] = python_action

            elif level == 2:

                level_0_action = self.execution_stack[0]["id"]

                current_policy = self.multilevel_policy[len(self.execution_stack)]["uber_action"]\
                [level_0_action]["policy"]
                torch_distribution = current_policy.critic_net.forward(torch_state)
                val, action = torch_distribution.max(0)
                level_1_action = action.item()

                # This is because the pre and post condition of the lowest level is same as total reward at final level
                previous_option = self.multilevel_policy[len(self.execution_stack)]["uber_action"]\
                [level_0_action]["uber_action"][level_1_action]
                # del previous_option["action"]

                previous_option["id"] = level_1_action

                # This policy network needs to be trained
                previous_option["policy"] = self.multilevel_policy[level][level_0_action][level_1_action]

                self.state_tracking = {}
                self.state_tracking["level_0_action"] = level_0_action
                self.state_tracking["level_1_action"] = level_1_action
                self.state_tracking["current_state_info"] = copy.deepcopy(obs)


                if previous_option["pre"](obs):
                    self.execution_stack.append(previous_option)
                else:
                    lower_option = self.multilevel_policy[len(self.execution_stack)]\
                    ["uber_action"][self.block_count]["uber_action"][self.lower_actions]
                    lower_option["policy"] = self.multilevel_policy[level][self.block_count][level_1_action]
                    lower_option["id"] = -1
                    self.execution_stack.append(lower_option)

            else:
                print("Uknown level - ", level, " exiting.. ")
                sys.exit()

        else :
            self.option_timeout[0] += 1
            self.option_timeout[1] += 1

            if level != 2:
                self.actual_action = self.execution_stack[-1]["action"](obs)
                if(self.execution_stack[-1]["post"](obs)) or (self.option_timeout[1] > 50):

                    # if (self.execution_stack[-1]["post"](obs)):
                    #     print("\t Ended because post condition satisfied")
                    # elif (self.option_timeout[1] > 50):
                    #     print("\t Ended due to timeout ")

                    if level == 1 and any(self.state_tracking) :
                        made_uber_transition = True
                        current_goal = obs['desired_goal'].astype(np.float32)
                        self.state_tracking["next_state_goal"] = torch_state
                        self.state_tracking["achieved_goal"] = current_goal

                    self.option_timeout[1] = 0
                    self.execution_stack.pop()

            elif self.state_tracking["level_0_action"] != self.block_count :

                self.state_tracking["next_state_info"] = copy.deepcopy(obs)
                self.state_tracking["action"] = self.actual_action

                self.actual_action, self.state_tracking = self.compute_continuous_action(self.state_tracking)


                made_uber_transition = True # At level 2 everything is uber action

            else:
                self.state_tracking = {}
                action = self.execution_stack[-1]["policy"].actor_net.forward(torch_state)
                self.actual_action = action.cpu().clone().detach().numpy()




        if made_uber_transition :
            trans_copy = copy.deepcopy(self.state_tracking)
            # print(" \t Made uber transition ")
            if level == 0 or self.execution_stack[-1]["post"](obs) or (self.option_timeout[0] > 80):
                trans_copy["terminal"] = True
            else:
                trans_copy["terminal"] = False


            if level == 2:
                current_id = self.execution_stack[-1]["id"]
                if self.execution_stack[-1]["post"](obs) :
                    trans_copy["terminal"] = True
                    trans_copy["reward"] = 0.0
                    self.option_timeout[1] = 0
                    self.execution_stack.pop()

                elif self.option_timeout[1] > 30 :
                    trans_copy["terminal"] = True
                    trans_copy["reward"] = perform_task.compute_custom_reward(trans_copy["desired_goal"], trans_copy["achieved_goal"])
                    self.option_timeout[1] = 0
                    self.execution_stack.pop()
                else:
                    trans_copy["terminal"] = False
                    trans_copy["reward"] = perform_task.compute_custom_reward(trans_copy["desired_goal"], trans_copy["achieved_goal"])

                self.state_tracking = {}
                self.state_tracking["level_0_action"] = trans_copy["level_0_action"]
                self.state_tracking["level_1_action"] = trans_copy["level_1_action"]

                self.state_tracking["current_state_info"] = copy.deepcopy(obs)

            else:
                self.state_tracking = {}

            if level == 2 and current_id == -1:
                trans_copy = None

            return trans_copy, self.actual_action

        return None, self.actual_action


    def slice_trajectory(self, trace, time_range, post_conditions, level = 0):

        trajectory_times = {}

        track_index_reward = {}
        index = 0
        target_list = []
        trace_slice = trace.copy()

        post_conds_list = list(post_conditions.keys())
        post_conds_list.sort(reverse = True)

        current_time = time_range[0]
        while (current_time in trace.keys()) and (current_time <= time_range[1]) and any(post_conds_list):
            current_state = trace_slice[current_time]['state']
            score = False

            each_cond = post_conds_list.pop()


            next_time, score = find_post_condition(trace_slice, current_time, post_conditions[each_cond])
            # print("Checking condition - ", each_cond, " score - ", score)

            if score:
                # print("At time - ", current_time, " for action - ", each_cond, " score - ", score, " next time stamp - ", next_time, \
                # " reward - ", trace_slice[next_time]['reward'])

                trajectory_times[each_cond] = [current_time, next_time]

                current_time = next_time

                if level == 0:
                    track_index_reward[index] = trace_slice[current_time]['reward']
                elif level == 1:
                    if len(post_conds_list) == 1:
                        track_index_reward[index] = trace_slice[current_time]['reward']
                        # track_index_reward[index] = 1
                    else:
                        track_index_reward[index] = 0

                index += 1
                target_list.append([current_state, each_cond, 0]) # This 0 is a placeholder
                # break

            if score:
                current_time += 1
                while current_time not in trace.keys() and current_time < 2 * len(trace.keys()) :
                    current_time += 1



        value = 0
        for index in reversed(sorted(track_index_reward.keys())):
            if value == 0:
                value = track_index_reward[index]
            else:
                value = track_index_reward[index] + self.gamma[level] * value
            target_list[index][2] = value

        # print("At level : ", level)
        # for each_tuple in target_list:
        #     print("For action - ", each_tuple[1], " value - ", each_tuple[2])

        return target_list, trajectory_times


    def save_to_disk(self, level = None):
        ''' Pick the policies and save them to their respective names. '''
        if level == 0 :
            self.multilevel_policy[level]["policy"].save_to_disk()
        elif level == 1:
            for each_key in self.multilevel_policy[level]["uber_action"].keys():
                self.multilevel_policy[level]["uber_action"][each_key]["policy"].save_to_disk()
        elif level == 2:
            for each_block in self.multilevel_policy[level].keys():
                if (type(each_block) is int) and each_block < self.block_count:
                    for each_action in self.multilevel_policy[level][each_block].keys():
                        self.multilevel_policy[level][each_block][each_action].save_weights()
        else:
            print("Level not implemented yet.")
            sys.exit()



def compute_reward(state, goal):
    if np.isclose(state, goal) :
        return 1.0
    else :
        return 0.0

def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - tau) + param.data * tau
        )

def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)



def assign_score_to_decision(trace, time_index, post_condition):

    current_state = trace[time_index]['state']
    uber_transition = estimate_uber_output(current_state, uber_action)
    time_stamp, score = find_if_state_in_trace(uber_transition, trace, time_index+1, uber_action)

    return time_stamp, score

def find_post_condition(trace, time_index, post_condition, thresh = 0.05):
    # init_unsat = False

    # if time_index not in trace.keys():
    #     return None, False

    for each_time_stamp in trace.keys():
        if each_time_stamp >= time_index :
            obs = trace[each_time_stamp]['state']
            if post_condition(obs):
                return each_time_stamp, True

    return None, False


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
