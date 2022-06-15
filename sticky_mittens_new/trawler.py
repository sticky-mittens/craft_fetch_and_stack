import numpy as np


print_anyway = True
folder = "logs"
m = {"gold": {
    "get[gold]": [],
    "make[bridge]": [],
    "get[iron]": [],
    "get[wood]": []
    }, 
    "gem": {
    "get[gem]": [], 
    "make[axe]": [],
    "make[stick]": [],
    "get[iron]": [],
    "get[wood]": []
    }}

def mean(l):
    return np.round(np.mean(np.array(l)),0)  
def std(l):
    return np.round(np.std(np.array(l)),0)   
def final(d):
    print('\n\n\n')
    for key in d.keys():
        print(key, '{} \scriptsize{{$\pm$ {}}}'.format(mean(d[key]), std(d[key])))

def process(store_eps, goal):
    print([k for k in store_eps.keys()])

    m[goal]['get[iron]'].append(store_eps['get[iron]'])
    m[goal]['get[wood]'].append(store_eps['get[wood]'])
    if goal == "gold":
        m[goal]['make[bridge]'].append(store_eps['get[iron]'] + store_eps['get[wood]'] + store_eps['make[bridge]'])
        m[goal]['get[gold]'].append(store_eps['get[iron]'] + store_eps['get[wood]'] + store_eps['make[bridge]'] + store_eps['get[gold]'])
    elif goal == "gem":
        m[goal]['make[stick]'].append(store_eps['get[wood]'] + store_eps['make[stick]'])
        m[goal]['make[axe]'].append(store_eps['get[wood]'] + store_eps['make[stick]'] + store_eps['get[iron]'] + store_eps['make[axe]'])
        m[goal]['get[gem]'].append(store_eps['get[wood]'] + store_eps['make[stick]'] + store_eps['get[iron]'] + store_eps['make[axe]'] + store_eps['get[gem]'])
    


seed_choices = {"gold": [1, 5, 10, 25, 50, 70, 80, 90, 100, 1000], "gem": [8, 10, 705, 905, 1575, 2050, 2100]} # 1225, 
total_eps_choices = {"gold": 20000, "gem": 25000}
for goal in ["gold", "gem"]:
    for seed in seed_choices[goal]:
        file = folder+"/"+"{}_{}_finetune_no_track.log".format(goal, seed)
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
            process(store_eps, goal)
        else:
            print('')


print(final(m['gold']))

print(final(m['gem']))