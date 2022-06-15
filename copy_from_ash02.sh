# rsync -azvP -e 'ssh' ksridhar@ash02.seas.upenn.edu:~/RL/ ./

# rsync -azvP -e 'ssh' ksridhar@ash02.seas.upenn.edu:~/RL/sticky_mittens_new/models/gem_8_finetune ./sticky_mittens_new/models/
# rsync -azvP -e 'ssh' ksridhar@ash02.seas.upenn.edu:~/RL/sticky_mittens_new/models/gem_10_finetune ./sticky_mittens_new/models/
# rsync -azvP -e 'ssh' ksridhar@ash02.seas.upenn.edu:~/RL/sticky_mittens_new/logs/gem_8* ./sticky_mittens_new/logs/
# rsync -azvP -e 'ssh' ksridhar@ash02.seas.upenn.edu:~/RL/sticky_mittens_new/logs/gem_10* ./sticky_mittens_new/logs/

# for i in 1 5 25 1000
# do 
#     rsync -azvP -e 'ssh' ksridhar@ash02.seas.upenn.edu:~/RL/sticky_mittens_new/models/gold_${i}_finetune ./sticky_mittens_new/models/
#     rsync -azvP -e 'ssh' ksridhar@ash02.seas.upenn.edu:~/RL/sticky_mittens_new/logs/gold_${i}* ./sticky_mittens_new/logs/
# done

rsync -azvP -e 'ssh' ksridhar@ash02.seas.upenn.edu:~/RL/sticky_mittens_new/models ./sticky_mittens_new/
rsync -azvP -e 'ssh' ksridhar@ash02.seas.upenn.edu:~/RL/sticky_mittens_bottom_up/models ./sticky_mittens_bottom_up/

rsync -azvP -e 'ssh' ksridhar@ash02.seas.upenn.edu:~/RL/fetch_sticky_mittens_step_4_blocks/networks ./fetch_sticky_mittens_step_4_blocks/
rsync -azvP -e 'ssh' ksridhar@ash02.seas.upenn.edu:~/RL/fetch_sticky_mittens_step/networks ./fetch_sticky_mittens_step/
