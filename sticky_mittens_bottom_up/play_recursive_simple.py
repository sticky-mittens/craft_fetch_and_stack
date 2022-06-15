from __future__ import division
from __future__ import print_function

import numpy as np
import time
import sys
import os
import yaml

import torch
import env_factory
import json
import learn_sub_task_finetune as learn_sub_task
import argparse

# Get the first level ingredients you would need for the current goal
# as an uber action called GIVE-> something.
# once it becomes possible to achieve that goal.
# Set the goal for the next round as the ingredients of the current round.
# Learn how to get the uber stuff
# and get the stuff one level underneath as uber actions.

steps = {'get[wood]':100, 'get[iron]':100, 'make[stick]':200, 'make[axe]':400, 'make[bridge]':300, 'get[gold]':400, 'get[gem]':500}

parser = argparse.ArgumentParser(description='Sticky Mittens to Uber Actions')
parser.add_argument('--topmost-goal', default='get[gem]', type=str, help='current goal: e.g. get[gold] or get[gem] (default: get[gold])')
parser.add_argument('--recipes-path', default='resources/recipes.yaml', type=str, help='path to recipes.yaml (default: resources/recipes.yaml)')
parser.add_argument('--hints-path', default='resources/hints.yaml', type=str, help='path to hints.yaml (default: resources/hints.yaml)')
parser.add_argument('--no-of-init-episodes', default=5000, type=int, help='number of initial episodes (default: 5000)')
parser.add_argument('--visualise', action='store_true', default=False, help='Add this NO-ARG flag if you want visualization output (default: False)')
parser.add_argument('--uid', default='gem_25', type=str, help='Unique id for this run')
parser.add_argument('--seed', default=25, type=int, help='random seed (default: 0)')
args = parser.parse_args()

for f in ['models', 'models/'+args.uid]:
    if not os.path.exists(f):
        os.mkdir(f)

# Seed
torch.manual_seed(5)

def get_all_goals(goal_name, hints_path): # new for bottom up: to just get list of all goals in hints.yaml without make0, mak1, make2 / makeAtWorkshop
    gem_full = ['get[wood]', 'make[stick]', 'get[iron]', 'make[axe]', 'get[gem]']
    gold_full = ['get[wood]', 'get[iron]', 'make[bridge]', 'get[gold]']
    matcher = {'get[gem]': gem_full, 'get[gold]': gold_full, 'make[axe]': gem_full[:-1], 'make[stick]': gem_full[:-3], 'make[bridge]': gold_full[:-1]}
    return matcher[goal_name]

# Get all goals
all_goals = get_all_goals(args.topmost_goal, args.hints_path)
print("All goals are: ", all_goals)

total_episodes = 0
for current_goal in all_goals:

    # Setup env
    env_sampler = env_factory.EnvironmentFactory(args.recipes_path, args.hints_path, max_steps=steps[current_goal], reuse_environments=False, visualise=args.visualise)
    env = env_sampler.sample_environment(task_name = args.topmost_goal)
    print("Environment: {} | Task: {}".format(env.task_name, env.task))
    
    print("Currently working on goal - ", current_goal)

    # Get all actions for current_goal
    ingredient_list = env.get_ingredients_for_goal(current_goal, args.hints_path)
    print("ingredient_list - ", ingredient_list)
    all_actions = env.uber_action_specs(ingredient_list)
    basic_actions = env.action_specs()
    uber_actions = {}
    for each_action in all_actions.keys():
        if each_action not in basic_actions.keys():
            uber_actions[each_action] = all_actions[each_action]
    print("Current actions - ", all_actions)
    
    avg_reward = 0
    while avg_reward < 0.8:
        # Reset the environment
        env.reset()
        
        avg_reward, calls_made, _ = learn_sub_task.learn(env, current_goal, \
        args.no_of_init_episodes, (all_actions, basic_actions, uber_actions), \
        args.uid, args.visualise)

        total_episodes += args.no_of_init_episodes
        print("Total episodes played till now - ", total_episodes)
