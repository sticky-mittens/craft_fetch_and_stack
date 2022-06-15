
SEED=50
mkdir fetch_logs

########### 3 blocks ###########
# cd fetch_sticky_mittens_step/train_policies
# CUDA_VISIBLE_DEVICES=2 nohup python -u train_flat_policy.py --no-of-blocks 3 --seed $SEED --folder-name baseline_block_3_uber_level_0_seed_$SEED --starting-level 0 --demos-file ../gen_demonstration_log/demos.p > ../../fetch_logs/baseline_step_block_3_uber_level_0_seed_$SEED.log &
# cd ../../

# cd fetch_sticky_mittens_sparse/train_policies
# CUDA_VISIBLE_DEVICES=2 nohup python -u train_flat_policy.py --seed $SEED --folder-name baseline_block_3_uber_level_0_seed_$SEED --starting-level 0 > ../../fetch_logs/baseline_sparse_block_3_uber_level_0_seed_$SEED.log &
# cd ../../

########### 4 blocks ##########

cd fetch_sticky_mittens_step_4_blocks/train_policies
CUDA_VISIBLE_DEVICES=2 nohup python -u train_flat_policy.py --no-of-blocks 4 --seed $SEED --folder-name baseline_block_4_uber_level_0_seed_$SEED --starting-level 0 --demos-file ../gen_demonstration_log/demos.p > ../../fetch_logs/baseline_step_block_4_uber_level_0_seed_$SEED.log &
cd ../../

# cd fetch_sticky_mittens_sparse_4_blocks/train_policies
# CUDA_VISIBLE_DEVICES=2 nohup python -u train_flat_policy.py --no-of-blocks 4 --seed $SEED --folder-name baseline_block_4_uber_level_0_seed_$SEED --starting-level 0 --demos-file ../gen_demonstration_log/demos.p > ../../fetch_logs/baseline_sparse_block_4_uber_level_0_seed_$SEED.log &
# cd ../../
