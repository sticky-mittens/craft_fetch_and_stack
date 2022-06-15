from __future__ import division
from __future__ import print_function

import numpy as np
import time
import sys
import os
import json

import torch
import env_factory
from output import policies_finetune
import learn_sub_task_no_track as learn_sub_task
import argparse


def play(env, main_policy_files, episodes, uid, visualise = False):

    sum_reward = 0
    count = 0
    for episode_index in range(episodes):
        done = False
        state = env.reset()

        basic_actions = env.action_specs()
        state = env.obs_specs()
        state_length = state['features']['shape'][0]

        execution_stack = []
        execution_stack.append(main_policy_files)

        # Find the actions corresponding to function calls
        actions = {}
        with open(execution_stack[-1]["info"], "r") as fp:
            actions = json.load(fp)

        execution_stack[-1]["action_info"] = actions
        policy = policies_finetune.Policy(state_length, 1, len(actions), \
        actor_file = execution_stack[-1]["actor"], critic_file = execution_stack[-1]["critic"] )
        execution_stack[-1]["policy"] = policy


        state = env.reset()

        t = 0
        reward = 0.
        while any(execution_stack) and (not done):

            # print("Current mode - ", execution_stack[-1]["termination"]) # commented this

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

                thing_name = learn_sub_task.get_thing_from_uber(action_name)

                if execution_stack[-1]["termination"] != thing_name and \
                (not env._sub_task_accomplished(thing_name)) :

                    # print("Calling uber action - ", action_name)

                    actor_network_file = "./models/"+uid+"/actor_" + thing_name + "_network.pkl"
                    critic_network_file = "./models/"+uid+"/critic_" + thing_name + "_network.pkl"
                    policy_info_file = "./models/"+uid+"/policy_info_" + thing_name + ".json"

                    assert os.path.exists(actor_network_file)
                    assert os.path.exists(critic_network_file)
                    assert os.path.exists(policy_info_file)

                    current_program = {"actor" : actor_network_file, "critic" : critic_network_file, \
                    "info": policy_info_file, "termination" : thing_name}

                    next_actions = {}
                    with open(policy_info_file, "r") as fp:
                        next_actions = json.load(fp)

                    current_program["action_info"] = next_actions

                    policy = policies_finetune.Policy(state_length, 1, len(next_actions), \
                    actor_file = current_program["actor"],\
                    critic_file = current_program["critic"])

                    current_program["policy"] = policy

                    execution_stack.append(current_program)



            # If the current goal is reached, remove it from the execution stack
            if env._sub_task_accomplished(execution_stack[-1]["termination"]) :
                ret = execution_stack.pop()


            t += 1

            if visualise:
              env.render_matplotlib(frame=state['image'])

            if reward:
                if visualise:
                  rewarding_frame = state['image'].copy()
                  rewarding_frame[:40] *= np.array([0, 1, 0])
                  env.render_matplotlib(frame=rewarding_frame, delta_time=0.7)
                #print("[{}] Got a rewaaaard! {:.1f}".format(t, reward)) # commented this and added below
                sum_reward += reward
                count += 1
                print("Episode: {} | [At t={}] Got reward {:.1f} | Avg Reward: {}".format(episode_index, t, reward, sum_reward/count))
            elif done:
                if visualise:
                  env.render_matplotlib(frame=np.zeros_like(state['image']), delta_time=0.3)
                count +=1
                #print("[{}] Finished with nothing... Reset".format(t)) # commented this and added below
                print("Episode: {} | [At t={}] Finished with nothing... Reset | Avg Reward: {}".format(episode_index, t, sum_reward/count))


def main():
    parser = argparse.ArgumentParser(description='Sticky Mittens to Uber Actions @ Test Time')
    parser.add_argument('--visualise', action='store_true', default=False, help='Add this NO-ARG flag if you want visualization output (default: False)')
    parser.add_argument('--current-goal', default='get[gold]', type=str, help='current goal: e.g. get[gold] or get[gem] (default: get[gold])')
    parser.add_argument('--current-thing', default='gold', type=str, help='current thing name: e.g. gold or gem (default: gold)')
    parser.add_argument('--uid', default='gold_1', type=str, help='Unique id for this run')
    parser.add_argument('--max-steps', default=400, type=int, help='maximum steps/states at test time (default: 400)')
    parser.add_argument('--no-of-test-episodes', default=100, type=int, help='number of episodes to test and avg reward (default: 100)')
    args = parser.parse_args()

    recipes_path = "resources/recipes.yaml"
    hints_path = "resources/hints.yaml"
    env_sampler = env_factory.EnvironmentFactory(
        recipes_path, hints_path, max_steps=args.max_steps, reuse_environments=False,
        visualise=args.visualise)

    actor_network_file = "./models/"+args.uid+"/actor_"+args.current_thing+"_network.pkl"
    critic_network_file = "./models/"+args.uid+"/critic_"+args.current_thing+"_network.pkl"
    policy_info_file = "./models/"+args.uid+"/policy_info_"+args.current_thing+".json"
    thing_name = args.current_thing

    policy_info = {"actor" : actor_network_file, "critic" : critic_network_file, \
    "info": policy_info_file, "termination" : thing_name}

    env = env_sampler.sample_environment(task_name=args.current_goal)
    print("Environment: task {}: {}".format(env.task_name, env.task))
    play(env, policy_info, args.no_of_test_episodes, args.uid, visualise=args.visualise)


if __name__ == '__main__':
  main()
