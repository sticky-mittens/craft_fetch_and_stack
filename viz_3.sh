export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so # for mujco error "GLEW initalization error: Missing GL version"

SEED=250

# First, ENSURE the below folder with all its networks is in ./fetch_sticky_mittens_step/networks/ directory
FOLDER=ours_10k_block_3_uber_level_0_seed_$SEED

cd fetch_sticky_mittens_step_4_blocks/train_policies # cd into 4 blocks folder but set blocks to 3 and use networks from 3 blocks and 3 blocks demos file!

FOLDER_LOC_FROM_INSIDE_4BLOCK_NETWORKS_DIR=../../fetch_sticky_mittens_step/networks/$FOLDER

# The _viz file calls the policies_viz file which does only exploiting (and no exploration) after loading the pretrained nets!
python -u train_hierarchical_with_uber_viz.py --visualise --no-of-blocks 3 --seed $SEED --folder-name $FOLDER_LOC_FROM_INSIDE_4BLOCK_NETWORKS_DIR --starting-level 0 --demos-file ../../fetch_sticky_mittens_step/gen_demonstration_log/demos.p
cd ../../

# You have to record by pressing V in mujoco!
# Note to change directory where video file will be saved, see: https://www.roboti.us/forum/index.php?threads/where-is-the-video-file-recorded-within-windows.3514/#post-5608 
