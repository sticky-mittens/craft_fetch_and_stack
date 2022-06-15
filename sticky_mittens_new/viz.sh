# Visualise training

python play_recursive_finetune_no_track.py --current-goal get[gold] --uid gold_50_finetune --seed 50 --visualise

python play_recursive_finetune_no_track.py --current-goal get[gem] --uid gem_50_finetune --seed 50 --visualise

# Visulise composed

python play_composed_craft.py --current-goal get[gold] --current-thing gold --uid gold_50_finetune --visualise

python play_composed_craft.py --current-goal get[gem] --current-thing gem --uid gem_8_finetune --visualise
