export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so # for mujco error "GLEW initalization error: Missing GL version"

SEED=150  

# First, ENSURE you have the below folder name and all its neural nets inside the ./fetch_sticky_mittens_step_4_blocks/networks/ directory
FOLDER=ours_10k_block_4_uber_level_0_seed_$SEED

cd fetch_sticky_mittens_step_4_blocks/train_policies
# The _viz file calls the policies_viz file which does only exploiting (and no exploration) after loading the pretrained nets!
python -u train_hierarchical_with_uber_viz.py --visualise --no-of-blocks 4 --seed $SEED --folder-name $FOLDER --starting-level 0 --demos-file ../gen_demonstration_log/demos.p
cd ../../

# You have to record by pressing V in mujoco!
# Note to change directory where video file will be saved, see: https://www.roboti.us/forum/index.php?threads/where-is-the-video-file-recorded-within-windows.3514/#post-5608 
