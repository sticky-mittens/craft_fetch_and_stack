import numpy as np
from copy import deepcopy as dc
import random
import sys
sys.path.append("../")
from gen_demonstration_log import perform_task

class Memory:
    def __init__(self, capacity, k_future):
        self.capacity = capacity
        self.memory = []
        self.memory_counter = 0
        self.memory_length = 0

        self.future_p = 1 - (1. / (1 + k_future))

    def sample(self, batch_size):


        max_length = -1
        # Compute the episode wise lengths first
        ep_wise_len = {}
        for ep_id in range(len(self.memory)):
            ep_wise_len[ep_id] = len(self.memory[ep_id]["next_state"])
            if ep_wise_len[ep_id] > max_length :
                max_length = ep_wise_len[ep_id]


        ep_indices = np.random.randint(0, len(self.memory), batch_size)

        time_indices = {}
        for ep_id in ep_indices:
            time_indices[ep_id] = np.random.randint(0, ep_wise_len[ep_id])


        states = []
        actions = []
        desired_goals = []
        next_states = []
        next_achieved_goals = []

        for episode in ep_indices:
            timestep = time_indices[episode]
            states.append(dc(self.memory[episode]["state"][timestep]))
            actions.append(dc(self.memory[episode]["action"][timestep]))
            desired_goals.append(dc(self.memory[episode]["desired_goal"][timestep]))
            next_achieved_goals.append(dc(self.memory[episode]["next_achieved_goal"][timestep]))
            next_states.append(dc(self.memory[episode]["next_state"][timestep]))


        states = np.vstack(states)
        actions = np.vstack(actions)
        desired_goals = np.vstack(desired_goals)
        next_achieved_goals = np.vstack(next_achieved_goals)
        next_states = np.vstack(next_states)

        future_times = {}
        ep_wise_HER = {}
        index = 0
        for episode in ep_indices:
            if np.random.uniform() < self.future_p :
                ep_wise_HER[episode] = np.random.randint(0, ep_wise_len[episode])
                future_offset = np.random.uniform() * ( ep_wise_len[episode] - time_indices[episode])
                future_offset = int(future_offset)
                future_t = time_indices[episode] + 1 + future_offset
                future_times[episode] = future_t
                desired_goals[index] = dc(self.memory[episode]["achieved_goal"][future_t])
            index += 1

        rewards = np.expand_dims(perform_task.compute_custom_reward(next_achieved_goals, desired_goals), 1)

        return self.clip_obs(states), actions, rewards, self.clip_obs(next_states), self.clip_obs(desired_goals)

    def add(self, transition):
        if len(transition) > 1 :
            self.memory.append(transition)
        if len(self.memory) > self.capacity:
            self.memory.pop(0)
        assert len(self.memory) <= self.capacity

    def __len__(self):
        return len(self.memory)

    @staticmethod
    def clip_obs(x):
        return np.clip(x, -200, 200)

    def sample_for_normalization(self, batch):

        new_batch = []
        for ep in batch:
            if len(ep["next_state"]) > 0:
                new_batch.append(ep)

        batch = new_batch

        max_length = -1
        # Compute the episode wise lengths first
        ep_wise_len = {}
        for ep_id in range(len(batch)):
            ep_wise_len[ep_id] = len(batch[ep_id]["next_state"])
            if ep_wise_len[ep_id] > max_length :
                max_length = ep_wise_len[ep_id]


        ep_indices = np.random.randint(0, len(batch), max_length)

        time_indices = {}
        for ep_id in ep_indices:
            time_indices[ep_id] = np.random.randint(0, ep_wise_len[ep_id])

        states = []
        desired_goals = []

        for episode in ep_indices:
            states.append(dc(batch[episode]["state"][time_indices[episode]]))
            desired_goals.append(dc(batch[episode]["desired_goal"][time_indices[episode]]))

        states = np.vstack(states)
        desired_goals = np.vstack(desired_goals)

        future_times = {}
        ep_wise_HER = {}
        index = 0
        for episode in ep_indices:
            if np.random.uniform() < self.future_p :
                ep_wise_HER[episode] = np.random.randint(0, ep_wise_len[episode])
                future_offset = np.random.uniform() * ( ep_wise_len[episode] - time_indices[episode])
                future_offset = int(future_offset)
                future_t = time_indices[episode] + 1 + future_offset
                future_times[episode] = future_t
                desired_goals[index] = dc(batch[episode]["achieved_goal"][future_t])
            index += 1

        return self.clip_obs(states), self.clip_obs(desired_goals)
