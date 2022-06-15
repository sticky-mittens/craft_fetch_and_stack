import sys
sys.path.append("../")
import gym
from gen_demonstration_log import perform_task
from gen_demonstration_log import clone_behavior
from uber_policies import action_policies
import policies
import pickle
import numpy as np
import os.path
from os import path

traces_file = "../gen_demonstration_log/demos.p"
blocks_count = 4


if path.exists(traces_file):
    with open(traces_file, 'rb') as fp:
        traces = pickle.load(fp)
else:
    traces = perform_task.generate_traces_multilevel(blocks_count, 200)
    with open(traces_file, 'wb') as fp:
        pickle.dump(traces, fp, protocol = pickle.HIGHEST_PROTOCOL)



# clone_behavior.train_network_for_traces("../networks/critic_network.pkl", traces, uber_actions)

# clone_behavior.train_network_for_lower_task(0, 0, traces, "../networks/actor_network_reach_object_0.pkl")
