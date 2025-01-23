#!/bin/sh
seed_max=1

for seed in `seq ${seed_max}`;
do
    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES=0 \
    python ../../src/main.py --config qmix --env-config sc2 --run masker \
    --env_args.map_name 1c3s5z --run_mode Test --runner episode --batch_size_run 1 \
    --checkpoint_path path_of_target_agents_model --saia_checkpoint_path path_of_AdapAM_model \
done