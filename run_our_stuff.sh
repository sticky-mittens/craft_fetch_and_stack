
SEED=25
mkdir fetch_logs
############# All Step #############

########### 3 blocks ###########
cd fetch_sticky_mittens_step/train_policies
nohup python -u train_hierarchical_with_uber.py --no-of-blocks 3 --seed $SEED --folder-name ours_block_3_uber_level_0_seed_$SEED --starting-level 0 --demos-file ../gen_demonstration_log/demos.p > ../../fetch_logs/ours_step_block_3_uber_level_0_seed_$SEED.log &
#CUDA_VISIBLE_DEVICES=2 nohup python -u train_hierarchical_with_uber.py --no-of-blocks 3 --seed $SEED --folder-name ours_block_3_uber_level_0_seed_$SEED --starting-level 0 --demos-file ../gen_demonstration_log/demos.p > ../../fetch_logs/ours_step_block_3_uber_level_0_seed_$SEED.log &
cd ../../

########### 4 blocks ##########

cd fetch_sticky_mittens_step_4_blocks/train_policies
nohup python -u train_hierarchical_with_uber.py --no-of-blocks 4 --seed $SEED --folder-name ours_block_4_uber_level_0_seed_$SEED --starting-level 0 --demos-file ../gen_demonstration_log/demos.p > ../../fetch_logs/ours_step_block_4_uber_level_0_seed_$SEED.log &
#CUDA_VISIBLE_DEVICES=2 nohup python -u train_hierarchical_with_uber.py --no-of-blocks 4 --seed $SEED --folder-name ours_block_4_uber_level_0_seed_$SEED --starting-level 0 --demos-file ../gen_demonstration_log/demos.p > ../../fetch_logs/ours_step_block_4_uber_level_0_seed_$SEED.log &
cd ../../
