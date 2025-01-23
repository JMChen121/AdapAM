#!/bin/sh
seed_max=1

for seed in `seq ${seed_max}`;
do
    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES=0 \
    python ../../src/main.py --config=vdn_gfootball --env-config=gfootball --run default \
    --env_args.map_name academy_3_vs_1_with_keeper --run_mode Test --runner episode --batch_size_run 1 \
    --checkpoint_path path_of_target_agents_model --saia_checkpoint_path path_of_AdapAM_model \
done