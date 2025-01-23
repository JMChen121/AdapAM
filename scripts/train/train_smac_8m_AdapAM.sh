#!/bin/sh
seed_max=1

for seed in `seq ${seed_max}`;
do
    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES=0 \
    python ../../src/main.py --config qmix --env-config sc2 --run AdapAM \
    --env_args.map_name 8m --run_mode Train --runner parallel --batch_size_run 8 \
    --checkpoint_path path_of_target_agents_model
done