from __future__ import division
from __future__ import print_function

import numpy as np
import time
import sys
import os

import torch
import env_factory
import json
import learn_sub_task_no_track
import argparse

# Get the first level ingredients you would need for the current goal
# as an uber action called GIVE-> something.
# once it becomes possible to achieve that goal.
# Set the goal for the next round as the ingredients of the current round.
# Learn how to get the uber stuff
# and get the stuff one level underneath as uber actions.

steps = {'get[wood]':100, 'get[iron]':100, 'make[stick]':200, 'make[axe]':400, 'make[bridge]':300, 'get[gold]':400, 'get[gem]':500}

parser = argparse.ArgumentParser(description='Sticky Mittens to Uber Actions')
parser.add_argument('--current-goal', default='get[gold]', type=str, help='current goal: e.g. get[gold] or get[gem] (default: get[gold])')
parser.add_argument('--recipes-path', default='resources/recipes.yaml', type=str, help='path to recipes.yaml (default: resources/recipes.yaml)')
parser.add_argument('--hints-path', default='resources/hints.yaml', type=str, help='path to hints.yaml (default: resources/hints.yaml)')
parser.add_argument('--no-of-init-episodes', default=5000, type=int, help='number of initial episodes (default: 5000)')
parser.add_argument('--learning-freq', default=20, type=int, help='learning frequency (default: 20)')
parser.add_argument('--visualise', action='store_true', default=False, help='Add this NO-ARG flag if you want visualization output (default: False)')
parser.add_argument('--uid', default='gold_1_finetune', type=str, help='Unique id for this run')
# parser.add_argument('--max-steps', default=100, type=int, help='maximum steps/states in each environment (default: 100)')
parser.add_argument('--seed', default=1, type=int, help='random seed (default: 0)')
args = parser.parse_args()

highest_goal = args.current_goal

model_folder = 'models/'+args.uid
call_logs_folder = 'call_logs/'+args.uid
for f in ['models', model_folder]:
    if not os.path.exists(f):
        os.mkdir(f)

torch.manual_seed(args.seed)

current_goals = []
# current_goals = ['make[axe]']
current_goals = [highest_goal]
# current_goals.append(highest_goal)
done = False
total_episodes = 0
while not done:
    print("List of current goals - ", current_goals)

    next_goals = []


    for current_goal in current_goals :
        ######### moved from above
        env_sampler = env_factory.EnvironmentFactory(
            args.recipes_path, args.hints_path, max_steps=steps[current_goal], reuse_environments=False,
            visualise=args.visualise)
        env = env_sampler.sample_environment(task_name = highest_goal)
        print("Environment: task {}: {}".format(env.task_name, env.task))
        ######### end

        print("Currently working on goal - ", current_goal)

        ingredient_list = env.get_ingredients_for_goal(current_goal, args.hints_path)
        print("ingredient_list - ", ingredient_list)

        all_actions = env.uber_action_specs(ingredient_list)
        basic_actions = env.action_specs()
        uber_actions = {}

        for each_action in all_actions.keys():
            if each_action not in basic_actions.keys():
                uber_actions[each_action] = all_actions[each_action]

        print("Current actions - ", all_actions)

        for each_thing in ingredient_list:
            next_goals.append(each_thing[1])

        print("Next goals for uber actions - ", next_goals)

        # Read the init positions from the disk for the current goal
        thing_name = learn_sub_task_no_track.get_thing_name(current_goal)


        current_goal_achieved = False
        while not current_goal_achieved :

            # Reset the environment
            env.reset()

            avg_reward, calls_made, _ = learn_sub_task_no_track.learn(env, highest_goal, current_goal,
            args.no_of_init_episodes, (all_actions, basic_actions, uber_actions), args.visualise, model_folder)

            total_episodes += args.no_of_init_episodes
            print("Total episodes played till now - ", total_episodes)

            if avg_reward > 0.8 :
                current_goal_achieved = True



    if not any(next_goals):
        done = True
    else:
        # current goals = uber actions used in this step
        current_goals = next_goals
