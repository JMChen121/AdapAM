#!/bin/sh
seed_max=1

for seed in `seq ${seed_max}`;
do
    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES=0 \
    python ../../src/main.py --config=vdn_gfootball --env-config=gfootball --run default \
    --env_args.map_name academy_counterattack_easy --run_mode Train --runner parallel --batch_size_run 32
done