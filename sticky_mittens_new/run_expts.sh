mkdir logs

# for i in 10 20 30 40 50 60 70 80 90 100
# do 
#     nohup python -u play_recursive_finetune_no_track.py --current-goal get[gold] --uid gold_${i}_finetune --seed ${i} > logs/gold_${i}_finetune_no_track.log &
# done

# for i in {105..1000..50}
for i in 1650 1675 1700 1725 1750 1775 2000 2025 2050 2075 2100 2125 2150 2175
do 
    nohup python -u play_recursive_finetune_no_track.py --current-goal get[gem] --uid gem_${i}_finetune --seed ${i} > logs/gem_${i}_finetune_no_track.log &
done
