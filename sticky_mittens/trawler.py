print_anyway = True

folder = "logs_correct"
seed_choices = {"gold": [1,5,25,30,40,45,50,55,125,625,500], "gem": [10,25,40,50,60,70,80,90,130,170,210,250,290]}
total_eps_choices = {"gold": 20000, "gem": 25000}
for goal in ["gold", "gem"]:
    for seed in seed_choices[goal]:
        file = folder+"/"+"{}_{}_finetune.log".format(goal, seed)
        print(file)
        good_file = True
        store_eps = {}
        line_num_no_issue = [1]
        continue_search = True
        with open(file, "r") as f:
            for i, line in enumerate(reversed(f.readlines())):
                if i ==0:
                    if "Total episodes played till now" not in line:
                        print("NOT COMPLETE!")
                    else:
                        total_eps = float(line.split()[-1])
                        if total_eps > total_eps_choices[goal]:
                            print(" ------------- Skipping this file as total episodes is {} ------------- ".format(total_eps))
                            good_file=False

                if continue_search and "At episode" in line:
                    ep = float(line.split()[3])
                    avg_reward = float(line.split()[-1])
                    if avg_reward < 0.8:
                        ep_store = ep + 20
                        continue_search = False
                elif "Currently working on goal" in line:
                    line_num_no_issue.append(i)
                    continue_search = True
                    task = line.split()[-1]
                    store_eps[task] = ep_store
                elif "When returning from function avg reward" in line:
                    if i not in line_num_no_issue:
                        ep_store += 5000
        if good_file or print_anyway:
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

m = {"get_gold": [4820+1780+800+2420, 1440+460+2260+980, 1740+840+4280+740, 780+3740+1260+2840, 1580+1480+1320+3640], # gold: 5, 25, 45, 50, 500
    "make bridge": [4820+1780+800, 1440+460+2260, 1740+840+4280, 780+3740+1260, 1580+1480+1320],
    "get iron for gold": [4820, 1440, 1740, 780, 1580],
    "get wood for gold": [1780, 460, 840, 3740, 1480],
    "get gem": [2940+1240+4960+4840+400, 880+720+3840+4980+340, 2360+600+4380+4980+420, 2340+400+1080+4920+380, 780+580+4840+4980+500], # gem: 10, 50, 90, 170, 290
    "make axe": [2940+1240+4960+4840, 880+720+3840+4980, 2360+600+4380+4980, 2340+400+1080+4920, 780+580+4840+4980],
    "make stick": [2940+4960, 880+3840, 2360+4380, 2340+1080, 780+4840],
    "get iron for gem": [1240, 720, 600, 400, 580],
    "get wood for gem": [2940, 880, 2360, 2340, 780],
    }

final(m)

     
