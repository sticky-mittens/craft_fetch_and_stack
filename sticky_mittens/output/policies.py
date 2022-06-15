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
        self.no_of_layers = 1
        self.no_of_hidden_units = 100 # 200 - grass
        super(policyNet, self).__init__()

        self.lin_trans = []

        self.lin_trans.append(nn.Linear(no_of_inputs, self.no_of_hidden_units))
        self.lin_trans.append(torch.nn.ReLU())
        for layer_index in range(self.no_of_layers-2):
            self.lin_trans.append(nn.Linear(self.no_of_hidden_units, self.no_of_hidden_units))
            self.lin_trans.append(torch.nn.ReLU())

        self.lin_trans.append(nn.Linear(self.no_of_hidden_units, no_of_outputs))
        self.lin_trans.append(torch.nn.ReLU())
        self.lin_trans.append(torch.nn.Softmax())

        self.model = torch.nn.Sequential(*self.lin_trans)

    def forward(self, x):
        y = self.model(x)
        return y

class criticNet(nn.Module):
    def __init__(self, no_of_inputs = 10, no_of_outputs = 5):
        self.no_of_layers = 1
        self.no_of_hidden_units = 100 # 200-grass
        super(criticNet, self).__init__()

        self.lin_trans = []

        self.lin_trans.append(nn.Linear(no_of_inputs, self.no_of_hidden_units))
        self.lin_trans.append(torch.nn.ReLU())
        for layer_index in range(self.no_of_layers-2):
            self.lin_trans.append(nn.Linear(self.no_of_hidden_units, self.no_of_hidden_units))
            self.lin_trans.append(torch.nn.ReLU())
        self.lin_trans.append(nn.Linear(self.no_of_hidden_units, no_of_outputs))
        self.lin_trans.append(torch.nn.ReLU())

        self.model = torch.nn.Sequential(*self.lin_trans)

    def forward(self, x):
        y = self.model(x)
        return y

class Policy():
    def __init__(self, input_count, action_count, action_types, actor_file, critic_file):
        self.actor_file = actor_file
        self.critic_file = critic_file
        # EDIT
        self.actor_learning_rate = 1e-3 #5e-3
        self.critic_learning_rate = 1e-3

        if (os.path.exists(self.actor_file) and os.path.exists(self.critic_file)):
            self.actor_net = torch.load(self.actor_file)
            self.critic_net = torch.load(self.critic_file)
            # print("Picked the networks from previous iterations.. ")
        else :
            self.actor_net = policyNet(input_count, action_types)
            self.actor_target_net = policyNet(input_count, action_types)

            # self.critic_net = criticNet(input_count + action_count, 1)
            self.critic_net = criticNet(input_count, 1)
            self.critic_target_net = criticNet(input_count + action_count, 1)
            print("Built new networks")

        self.optimizer_actor = optim.Adam(self.actor_net.parameters(), lr = self.actor_learning_rate)
        self.optimizer_critic = optim.Adam(self.critic_net.parameters(), lr = self.critic_learning_rate)



    def update_policy(self, trajectories):

        if not trajectories:
            return

        # Estimate the return from the next states as learnt from the trajectories
        discounted_rewards = compute_return(trajectories)


        no_of_training_iterations = 1
        batch_size = len(trajectories)

        for training_step in range(no_of_training_iterations):
            # Build a batch of traning data
            episode_indices = list(trajectories.keys())
            actor_loss_list = []
            critic_loss_list = []
            entropy_loss_list = []

            self.optimizer_actor.zero_grad()
            self.optimizer_critic.zero_grad()

            # for batch_index in range(batch_size):
            for episode_index in episode_indices:
                time_indices = list(trajectories[episode_index].keys())

                for time_index in time_indices:

                    t = 0

                    if(trajectories[episode_index][time_index]['is_uber']):
                        t = torch.tensor(1.0, requires_grad = False)
                    else:
                        t = torch.tensor(1.0, requires_grad = False)

                    advantage = t * trajectories[episode_index][time_index]['QValue'] \
                    - torch.tensor(discounted_rewards[episode_index][time_index])


                    action_log = trajectories[episode_index][time_index]['action_log']
                    actor_loss_list.append(action_log * advantage.clone().detach())
                    critic_loss_list.append(0.5 * advantage.pow(2))

            # Take a gradient step
            # actor_loss = torch.stack(actor_loss_list, dim = 0).mean()
            # critic_loss = torch.stack(critic_loss_list, dim = 0).mean()
            actor_loss = torch.stack(actor_loss_list, dim = 0).sum()
            critic_loss = torch.stack(critic_loss_list, dim = 0).sum()

            # print("Actor loss - ", actor_loss)
            # print("Critic loss - ", critic_loss)


            actor_loss.backward()
            critic_loss.backward()
            self.optimizer_actor.step()
            self.optimizer_critic.step()

    def save_to_disk(self):
        torch.save(self.actor_net, self.actor_file)
        torch.save(self.critic_net, self.critic_file)
