#nohup python -u play_composed_craft.py --current-goal get[gold] --current-thing gold --uid gold_1 --max-steps 400 > logs/gold_1_test.log &

#nohup python -u play_composed_craft.py --current-goal get[gem] --current-thing gem --uid gem_1 --max-steps 500 > logs/gem_1_test.log &

#nohup python -u play_composed_craft.py --current-goal get[gem] --current-thing gem --uid gem_1 --max-steps 1000 > logs/gem_1_test_1000steps.log &

########### new finetuned

# nohup python -u play_composed_craft.py --current-goal get[gold] --current-thing gold --uid gold_1_finetune --max-steps 400 > logs/gold_1_finetune_test.log &

# nohup python -u play_composed_craft.py --current-goal get[gold] --current-thing gold --uid gold_10_finetune --max-steps 400 > logs/gold_10_finetune_test.log &
# nohup python -u play_composed_craft.py --current-goal get[gold] --current-thing gold --uid gold_20_finetune --max-steps 400 > logs/gold_20_finetune_test.log &
# nohup python -u play_composed_craft.py --current-goal get[gold] --current-thing gold --uid gold_25_finetune --max-steps 400 > logs/gold_25_finetune_test.log &
# nohup python -u play_composed_craft.py --current-goal get[gold] --current-thing gold --uid gold_30_finetune --max-steps 400 > logs/gold_30_finetune_test.log &
# nohup python -u play_composed_craft.py --current-goal get[gold] --current-thing gold --uid gold_35_finetune --max-steps 400 > logs/gold_35_finetune_test.log &
# nohup python -u play_composed_craft.py --current-goal get[gold] --current-thing gold --uid gold_40_finetune --max-steps 400 > logs/gold_40_finetune_test.log &

nohup python -u play_composed_craft.py --current-goal get[gem] --current-thing gem --uid gem_25_finetune --max-steps 500 > logs/gem_25_finetune_test.log &
nohup python -u play_composed_craft.py --current-goal get[gem] --current-thing gem --uid gem_40_finetune --max-steps 500 > logs/gem_40_finetune_test.log &



# viz
# python play_composed_craft.py --current-goal get[gem] --current-thing gem --uid gem_25_finetune --max-steps 500 --visualise
