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

class Option_Collection():

    def __init__(self, no_of_blocks, folder_name, demos_file = "../gen_demonstration_log/demos.p"):
        self.block_count = no_of_blocks
        self.lower_actions = 3
        self.decay_rate = 1

        self.multilevel_policy = perform_task.get_flat_policy(no_of_blocks, folder_name)

        self.execution_stack = []
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.traces = {}

        self.demos_file = demos_file
        if path.exists(self.demos_file):
            with open(self.demos_file, 'rb') as fp:
                self.traces = pickle.load(fp)


        self.last_count = 0
        self.gamma = 0.9
        self.episode_count = 0
        self.option_timeout = 0
        self.lower_action_dimension = 4 # Sorry for hard-coding this
        self.book_keeping_reset()



    def book_keeping_reset(self):
        self.execution_stack = []
        self.option_timeout = 0
        self.state_tracking = {}
        self.actual_action = np.zeros(self.lower_action_dimension)


    def update_policy(self, exp_collection):

        if not exp_collection:
            return

        criterion = nn.SmoothL1Loss()

        batch_size = math.ceil(0.2 * len(exp_collection)) # 0:0.5

        current_policy = None
        current_action_length = len(self.multilevel_policy) - 1

        critic_loss_list = []


        # Compute the loss for each policy
        transition_collection = []

        for index in range(batch_size):

            trans = random.choice(list(exp_collection.values()))

            current_policy = self.multilevel_policy["policy"]
            transition_collection.append(trans)

            if trans['terminal'] :
                target_val = trans['reward']
            else:
                next_val, next_action = current_policy.critic_net.forward(trans['next_state_goal']).max(0)
                target_val = trans['reward'] + self.gamma * next_val.clone().detach()


            prediction = current_policy.critic_net.forward(trans['current_state_goal'])
            current_val, current_action = prediction.max(0)

            np_target = np.zeros(current_action_length)
            np_target[trans['action']] = target_val
            np_target = np_target.astype(np.float32)
            torch_target = torch.from_numpy(np_target).to(self.device)

            # print(" train Target - ", torch_target, " block no - ", top_action)

            loss = criterion(prediction, torch_target).to(self.device)

            critic_loss_list.append(loss)


        current_policy = self.multilevel_policy["policy"]
        current_policy.optimizer_critic.zero_grad()
        critic_loss = torch.stack(critic_loss_list, dim = 0).sum()
        critic_loss.backward()
        current_policy.optimizer_critic.step()



    def execute_policy(self, obs, episode_index):


        # print("Executing policy---- , length of execution stack - ", len(self.execution_stack))
        made_uber_transition = False
        current_state = obs['observation'].astype(np.float32)
        current_goal = obs['desired_goal'].astype(np.float32)
        achieved_goal = obs['achieved_goal'].astype(np.float32)
        torch_state = torch.cat((torch.tensor(current_state), torch.tensor(current_goal)))
        torch_state = torch_state.to(self.device)
        self.actual_action = np.zeros(self.lower_action_dimension)


        if len(self.execution_stack) == 0:

            exploration_number = np.exp(-1 * self.decay_rate * 1e-3 * \
            (self.episode_count + 1)) # 0.2

            if episode_index > self.last_count :
                self.last_count = episode_index
                self.episode_count += 1


            if np.random.uniform(0,1) <= exploration_number:
                action = np.random.randint(0, str(len(self.multilevel_policy)-3))
                python_action = action
            else:
                current_policy = self.multilevel_policy["policy"]
                torch_distribution = current_policy.critic_net.forward(torch_state)
                val, action = torch_distribution.max(0)
                # print("Action - ", action)
                python_action = action.item()

            # Note this assignment is only for the pre post condition checking not for execution
            current_option = self.multilevel_policy["option_" + str(python_action)]
            current_option["id"] = python_action

            # print("Action chosen at level 0 : ", python_action)

            if current_option["pre"](obs):
                self.execution_stack.append(current_option)
            else:
                # print("Choosing null because pre condition unsat")
                null_option = self.multilevel_policy["option_" + str(len(self.multilevel_policy)-2)]
                null_option["id"] = len(self.multilevel_policy)
                self.execution_stack.append(null_option)

            # Track transitions for level 0
            if not any(self.state_tracking):
                self.state_tracking["current_state_goal"] = torch_state
                self.state_tracking["action"] = python_action


        else :
            self.option_timeout += 1

            self.actual_action = self.execution_stack[-1]["action"](obs)
            if(self.execution_stack[-1]["post"](obs)) or (self.option_timeout > 50):

                if any(self.state_tracking) :
                    made_uber_transition = True
                    current_goal = obs['desired_goal'].astype(np.float32)
                    self.state_tracking["next_state_goal"] = torch_state
                    self.state_tracking["achieved_goal"] = current_goal
                    self.state_tracking['terminal'] = False

                    self.option_timeout = 0
                    self.execution_stack.pop()

                    made_uber_transition = True

            # else:
            #     self.state_tracking = {}
            #     action = self.execution_stack[-1]["policy"].actor_net.forward(torch_state)
            #     self.actual_action = action.cpu().clone().detach().numpy()


        if made_uber_transition :
            trans_copy = copy.deepcopy(self.state_tracking)
            self.state_tracking = {}
            return trans_copy, self.actual_action

        return None, self.actual_action


    def save_to_disk(self):
        self.multilevel_policy["policy"].save_to_disk()
