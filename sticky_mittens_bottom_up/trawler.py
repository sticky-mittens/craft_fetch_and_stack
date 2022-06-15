print_anyway = True


folder = "logs"
seed_choices = {"gold": [1,40,45,50,1000,1005,1010,1020,1025,1030,1035,1040], # gold: 45, 50, 1020, 1025, 1000, 1005,   1, 1035
                "gem": [10,25,40,50,60,70,80,1000,1005,1010,1020,1025,1030,1035,1040] # gem: 25, 50, 1005, 1040, 1035,   70
                }
total_eps_choices = {"gold": 20000, "gem": 25000}
for goal in ["gold", "gem"]:
    for seed in seed_choices[goal]:
        file = folder+"/"+"{}_{}.log".format(goal, seed)
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
                            #print(" ------------- Skipping this file as total episodes is {} ------------- ".format(total_eps))
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
    return np.round(np.mean(np.array(l)),2)  
def std(l):
    return np.round(np.std(np.array(l)),2)   
def final(d):
    print('\n\n\n')
    for key in d.keys():
        print(key, '{} $\pm$ {}'.format(mean(d[key]), std(d[key])))

m = {"get gem": [50000+9880+1220+1240+620], 
    "make axe": [9940+780+920+940, 14960+1520+4980+940, 4940+1480+980+1540, 1360+1260+1400+900, 9880+1220+1240+620],
    "make stick": [920+940, 4980+940, 980+1540, 1400+900, 1240+620],
    "get_gold": [9480+2920+1820+840, 4780+19720+1860+1420],
    "make bridge": [2920+1820+840, 19720+1860+1420]}    

final(m)