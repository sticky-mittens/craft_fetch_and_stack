mkdir logs

############ _400gold_500gem

# nohup python -u play_recursive_finetune.py --current-goal get[gold] --uid gold_1_finetune --seed 1--max-steps 400 > logs/gold_1_finetune.log & 
# nohup python -u play_recursive_finetune.py --current-goal get[gold] --uid gold_5_finetune --seed 5--max-steps 400 > logs/gold_5_finetune.log & 
# nohup python -u play_recursive_finetune.py --current-goal get[gold] --uid gold_25_finetune --seed 25--max-steps 400 > logs/gold_25_finetune.log & 
# nohup python -u play_recursive_finetune.py --current-goal get[gold] --uid gold_30_finetune --seed 30--max-steps 400 > logs/gold_30_finetune.log & 
# nohup python -u play_recursive_finetune.py --current-goal get[gold] --uid gold_40_finetune --seed 40--max-steps 400 > logs/gold_40_finetune.log & 
# nohup python -u play_recursive_finetune.py --current-goal get[gold] --uid gold_45_finetune --seed 45--max-steps 400 > logs/gold_45_finetune.log & 
# nohup python -u play_recursive_finetune.py --current-goal get[gold] --uid gold_50_finetune --seed 50--max-steps 400 > logs/gold_50_finetune.log & 
# nohup python -u play_recursive_finetune.py --current-goal get[gold] --uid gold_55_finetune --seed 55--max-steps 400 > logs/gold_55_finetune.log & 


# nohup python -u play_recursive_finetune.py --current-goal get[gem] --uid gem_25_finetune --seed 25--max-steps 500 > logs/gem_25_finetune.log & # good #########
# nohup python -u play_recursive_finetune.py --current-goal get[gem] --uid gem_40_finetune --seed 40--max-steps 500 > logs/gem_40_finetune.log & # good ##########
# nohup python -u play_recursive_finetune.py --current-goal get[gem] --uid gem_10_finetune --seed 10--max-steps 500 > logs/gem_10_finetune.log & # 
# nohup python -u play_recursive_finetune.py --current-goal get[gem] --uid gem_50_finetune --seed 50--max-steps 500 > logs/gem_50_finetune.log & # 
# nohup python -u play_recursive_finetune.py --current-goal get[gem] --uid gem_60_finetune --seed 60--max-steps 500 > logs/gem_60_finetune.log & # 
# nohup python -u play_recursive_finetune.py --current-goal get[gem] --uid gem_70_finetune --seed 70--max-steps 500 > logs/gem_70_finetune.log & # 
# nohup python -u play_recursive_finetune.py --current-goal get[gem] --uid gem_80_finetune --seed 80--max-steps 500 > logs/gem_80_finetune.log & # 
# nohup python -u play_recursive_finetune.py --current-goal get[gem] --uid gem_90_finetune --seed 90--max-steps 500 > logs/gem_90_finetune.log & # 

############ adaptive with different steps for different sub-tasks

# nohup python -u play_recursive_finetune.py --current-goal get[gold] --uid gold_1_finetune --seed 1 > logs/gold_1_finetune.log & 
# nohup python -u play_recursive_finetune.py --current-goal get[gold] --uid gold_5_finetune --seed 5 > logs/gold_5_finetune.log & 
# nohup python -u play_recursive_finetune.py --current-goal get[gold] --uid gold_25_finetune --seed 25 > logs/gold_25_finetune.log & 
# nohup python -u play_recursive_finetune.py --current-goal get[gold] --uid gold_30_finetune --seed 30 > logs/gold_30_finetune.log & 
# nohup python -u play_recursive_finetune.py --current-goal get[gold] --uid gold_40_finetune --seed 40 > logs/gold_40_finetune.log & 
# nohup python -u play_recursive_finetune.py --current-goal get[gold] --uid gold_45_finetune --seed 45 > logs/gold_45_finetune.log & 
# nohup python -u play_recursive_finetune.py --current-goal get[gold] --uid gold_50_finetune --seed 50 > logs/gold_50_finetune.log & 
# nohup python -u play_recursive_finetune.py --current-goal get[gold] --uid gold_55_finetune --seed 55 > logs/gold_55_finetune.log & 

nohup python -u play_recursive_finetune.py --current-goal get[gold] --uid gold_125_finetune --seed 125 > logs/gold_125_finetune.log & 
nohup python -u play_recursive_finetune.py --current-goal get[gold] --uid gold_625_finetune --seed 625 > logs/gold_625_finetune.log & 
nohup python -u play_recursive_finetune.py --current-goal get[gold] --uid gold_500_finetune --seed 500 > logs/gold_500_finetune.log & 
nohup python -u play_recursive_finetune.py --current-goal get[gold] --uid gold_1000_finetune --seed 1000 > logs/gold_1000_finetune.log & 


# nohup python -u play_recursive_finetune.py --current-goal get[gem] --uid gem_25_finetune --seed 25 > logs/gem_25_finetune.log & # good #########
# nohup python -u play_recursive_finetune.py --current-goal get[gem] --uid gem_40_finetune --seed 40 > logs/gem_40_finetune.log & # good ##########
# nohup python -u play_recursive_finetune.py --current-goal get[gem] --uid gem_10_finetune --seed 10 > logs/gem_10_finetune.log & # 
# nohup python -u play_recursive_finetune.py --current-goal get[gem] --uid gem_50_finetune --seed 50 > logs/gem_50_finetune.log & # 
# nohup python -u play_recursive_finetune.py --current-goal get[gem] --uid gem_60_finetune --seed 60 > logs/gem_60_finetune.log & # 
# nohup python -u play_recursive_finetune.py --current-goal get[gem] --uid gem_70_finetune --seed 70 > logs/gem_70_finetune.log & # 
# nohup python -u play_recursive_finetune.py --current-goal get[gem] --uid gem_80_finetune --seed 80 > logs/gem_80_finetune.log & # 
# nohup python -u play_recursive_finetune.py --current-goal get[gem] --uid gem_90_finetune --seed 90 > logs/gem_90_finetune.log & # 

nohup python -u play_recursive_finetune.py --current-goal get[gem] --uid gem_130_finetune --seed 130 > logs/gem_130_finetune.log & # 
nohup python -u play_recursive_finetune.py --current-goal get[gem] --uid gem_170_finetune --seed 170 > logs/gem_170_finetune.log & # 
nohup python -u play_recursive_finetune.py --current-goal get[gem] --uid gem_210_finetune --seed 210 > logs/gem_210_finetune.log & # 
nohup python -u play_recursive_finetune.py --current-goal get[gem] --uid gem_250_finetune --seed 250 > logs/gem_250_finetune.log & # 
nohup python -u play_recursive_finetune.py --current-goal get[gem] --uid gem_290_finetune --seed 290 > logs/gem_290_finetune.log & # 