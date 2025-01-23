#!/bin/sh
seed_max=1

for seed in `seq ${seed_max}`;
do
    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES=0 \
    python ../../src/main.py --config=vdn_gfootball --env-config=gfootball --run masker \
    --env_args.map_name academy_counterattack_easy --run_mode Test --runner episode --batch_size_run 1 \
    --checkpoint_path path_of_target_agents_model --saia_checkpoint_path path_of_AdapAM_model \
done