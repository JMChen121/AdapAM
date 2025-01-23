#!/usr/bin/env python
import os
import sys
from pathlib import Path

import numpy as np
import setproctitle
import torch

from config.mpe_conf import get_config
from envs.env_wrappers import SubprocVecEnv, DummyVecEnv
from envs.mpe.MPE_env import MPEEnv


def make_render_env(all_args):
    def get_env_fn(rank):
        def init_env():
            if all_args.env_name == "MPE":
                env = MPEEnv(all_args)
            else:
                print("Can not support the " +
                      all_args.env_name + "environment.")
                raise NotImplementedError
            env.seed(all_args.seed + rank * 1000)
            return env

        return init_env

    if all_args.n_rollout_threads == 1:
        return DummyVecEnv([get_env_fn(0)])
    else:
        return SubprocVecEnv([get_env_fn(i) for i in range(all_args.n_rollout_threads)])


def parse_args(args, parser):
    parser.add_argument('--scenario_name', type=str,
                        default='simple_spread', help="Which scenario to run on")
    parser.add_argument("--num_landmarks", type=int, default=3)
    parser.add_argument('--num_agents', type=int,
                        default=2, help="number of players")

    all_args = parser.parse_known_args(args)[0]

    return all_args


def main(args):
    parser = get_config()
    all_args = parse_args(args, parser)

    all_args.save_gifs = False
    all_args.share_policy = True
    all_args.env_name = "MPE"
    all_args.algorithm_name = "rmappo"
    all_args.experiment_name = f"{all_args.env_name}_{all_args.algorithm_name}_eval"
    all_args.seed = 1                              # np.random.randint(1, 100)
    all_args.n_training_threads = 1
    all_args.n_rollout_threads = 1
    all_args.episode_length = 100
    all_args.render_episodes = 500
    all_args.use_render = False
    all_args.cuda = False
    '''Evaluation Params'''
    all_args.agent_selection = "masker"         # masker, value_max, random, masker_, value_min
    all_args.need_RRD = False
    all_args.need_attack = False
    all_args.need_patch = True
    all_args.need_collect = False
    all_args.patch_budget = 50

    '''For MPE-Reference'''
    # all_args.scenario_name = "simple_reference"
    # all_args.num_agents = 2
    # all_args.num_landmarks = 3
    # all_args.run_policy = "masker"
    # all_args.masker_reward = 0.0
    # all_args.model_dir = "xxx"
    # all_args.masker_model_dir = "xxx"
    # all_args.ep_reward_th = -25
    # all_args.diff_th = 0.5              # 0.5

    '''For MPE-Spread'''
    all_args.scenario_name = "simple_spread"
    all_args.num_agents = 3
    all_args.num_landmarks = 3
    all_args.run_policy = "masker"
    all_args.masker_reward = 0.01
    all_args.model_dir = "xxx"
    all_args.masker_model_dir = "xxx"
    all_args.ep_reward_th = -520
    all_args.diff_th = 0.5                 # 0.5

    if all_args.algorithm_name == "rmappo":
        print("u are choosing to use rmappo, we set use_recurrent_policy to be True")
        all_args.use_recurrent_policy = True
        all_args.use_naive_recurrent_policy = False
    else:
        raise NotImplementedError

    assert (all_args.share_policy == True and all_args.scenario_name == 'simple_speaker_listener') == False, (
        "The simple_speaker_listener scenario can not use shared policy. Please check the mpe_conf.py.")

    # assert all_args.use_render, ("u need to set use_render be True")
    assert not (all_args.model_dir == None or all_args.model_dir == ""), ("set model_dir first")
    assert all_args.n_rollout_threads == 1, ("only support to use 1 env to render.")

    # cuda
    if all_args.cuda and torch.cuda.is_available():
        print("choose to use gpu...")
        device = torch.device("cuda:0")
        torch.set_num_threads(all_args.n_training_threads)
        if all_args.cuda_deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    else:
        print("choose to use cpu...")
        device = torch.device("cpu")
        torch.set_num_threads(all_args.n_training_threads)

    # run dir
    run_dir = Path(os.path.split(os.path.dirname(os.path.abspath(__file__)))[
                       0] + "/results") / all_args.env_name / all_args.scenario_name / all_args.algorithm_name / all_args.experiment_name
    if not run_dir.exists():
        os.makedirs(str(run_dir))

    if not run_dir.exists():
        curr_run = 'run1'
    else:
        exst_run_nums = [int(str(folder.name).split('run')[1]) for folder in run_dir.iterdir() if
                         str(folder.name).startswith('run')]
        if len(exst_run_nums) == 0:
            curr_run = 'run1'
        else:
            curr_run = 'run%i' % (max(exst_run_nums) + 1)
    run_dir = run_dir / curr_run
    if not run_dir.exists():
        os.makedirs(str(run_dir))

    setproctitle.setproctitle(str(all_args.algorithm_name) + "-" + \
                              str(all_args.env_name) + "-" + str(all_args.experiment_name) + "@" + str(
        all_args.user_name))

    # seed
    torch.manual_seed(all_args.seed)
    torch.cuda.manual_seed_all(all_args.seed)
    np.random.seed(all_args.seed)

    # env init
    envs = make_render_env(all_args)
    eval_envs = None
    num_agents = all_args.num_agents

    config = {
        "all_args": all_args,
        "envs": envs,
        "eval_envs": eval_envs,
        "num_agents": num_agents,
        "device": device,
        "run_dir": run_dir
    }

    # run experiments
    if all_args.share_policy:
        if all_args.run_policy == "masker":
            from runners.shared.mpe_runner_masker import MPERunnerMasker as Runner
        else:
            from runners.shared.mpe_runner import MPERunner as Runner
    else:
        from runners.separated.mpe_runner import MPERunner as Runner
    runner = Runner(config)

    if all_args.need_collect:
        runner.run_single_exploration()
    else:
        if all_args.need_patch:
            runner.load_patch()
        mean_episode_rewards = []
        mean_attack_rate = []
        mean_patch_rate = []
        run_times = 2 if all_args.need_patch else 3
        for i in range(run_times):
            mean_episode_reward = runner.render()
            mean_episode_rewards.append(mean_episode_reward)
            if all_args.need_attack:
                mean_attack_rate.append(runner.attack_valid / (runner.attack_valid + runner.attack_invalid + 1e-5))
            if all_args.need_patch:
                mean_patch_rate.append(runner.patch_valid / (runner.patch_valid + runner.patch_invalid + 1e-5))

        end_info_str = f"End Info: \n" \
                       f"Episode rewards: {mean_episode_rewards}; Average: {np.mean(mean_episode_rewards)}; STD: {np.std(mean_episode_rewards)}\n"
        if all_args.need_attack:
            end_info_str += f"Attack rates: {mean_attack_rate}; Average: {np.mean(mean_attack_rate)}; STD: {np.std(mean_attack_rate)}\n"
        if all_args.need_patch:
            end_info_str += f"Patch rates: {mean_patch_rate}; Average: {np.mean(mean_patch_rate)}; STD: {np.std(mean_patch_rate)}\n"
        print(end_info_str)

    # post process
    envs.close()


if __name__ == "__main__":
    main(sys.argv[1:])
