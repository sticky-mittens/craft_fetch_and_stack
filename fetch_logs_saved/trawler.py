print_anyway = True

folder = "."
seed_choices = {"step_block_3": [25, 50, 100], "step_block_4": [25, 50, 100]}
total_eps_choices = {"step_block_3": 3000, "step_block_4": 3000}
timesteps_choices = {"step_block_3": 150, "step_block_4": 200}
for top_goal in ["step_block_3", "step_block_4"]:
    for seed in seed_choices[top_goal]:
        file = folder+"/"+"ours_{}_uber_level_0_seed_{}.log".format(top_goal, seed)
        print(file)
        store_eps = {}
        continue_search = True
        with open(file, "r") as f:
            for i, line in enumerate(reversed(f.readlines())):
                if continue_search and ("At episode" in line and "Average reward" in line):
                    ep = float(line.split()[3])
                    t = ep * timesteps_choices[top_goal]
                    avg_reward = float(line.split()[-1])
                    if avg_reward < 0.8:
                        ep_store = ep + 20
                        continue_search = False
                elif "Learning at level" in line:
                    level = line.split()[-1]
                    continue_search = True
                    if float(level) == 2:
                        continue
                    else:
                        store_eps[level] = ep_store
        if print_anyway:
            print(store_eps)
        else:
            print('')

import numpy as np
def mean(l):
    return np.round(np.mean(np.array(l)),0)  
def std(l):
    return np.round(np.std(np.array(l)),0)   
def final(d):
    print('\n\n\n')
    for key in d.keys():
        print(key, '{} \scriptsize{{$\pm$ {}}}'.format(mean(d[key]), std(d[key])))

m = {
    "3 blocks step": [0],
    "4 blocks step": [0]
}

final(m)

     
