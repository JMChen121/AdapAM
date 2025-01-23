import datetime
import os
import pprint
import time
import threading

import numpy as np
import torch as th
from types import SimpleNamespace as SN
from utils.logging import Logger
from utils.timehelper import time_left, time_str
from os.path import dirname, abspath

from learners import REGISTRY as le_REGISTRY
from learners import SAIA_REGISTRY as saia_le_REGISTRY
from runners import SAIA_REGISTRY as saia_r_REGISTRY
from controllers import REGISTRY as mac_REGISTRY
from controllers import SAIA_REGISTRY as saia_mac_REGISTRY
from components.episode_buffer import ReplayBuffer
from components.transforms import OneHot

from smac.env import StarCraft2Env


def get_agent_own_state_size(env_args):
    sc_env = StarCraft2Env(**env_args)
    # qatten parameter setting (only use in qatten)
    return 4 + sc_env.shield_bits_ally + sc_env.unit_type_bits


def run(_run, _config, _log):
    # check args sanity
    _config = args_sanity_check(_config, _log)

    args = SN(**_config)
    saia_args = SN(**_config)
    args.device = "cuda" if args.use_cuda else "cpu"

    # setup loggers
    logger = Logger(_log)

    _log.info("Experiment Parameters:")
    experiment_params = pprint.pformat(_config,
                                       indent=4,
                                       width=1)
    _log.info("\n\n" + experiment_params + "\n")

    # configure tensorboard logger
    unique_token = "{}__{}".format(args.name, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    args.unique_token = unique_token
    saia_args.unique_token = unique_token
    if args.use_tensorboard:
        tb_logs_direc = os.path.join(dirname(dirname(dirname(abspath(__file__)))), "results", "tb_logs")
        tb_exp_direc = os.path.join(tb_logs_direc, "{}").format(unique_token)
        logger.setup_tb(tb_exp_direc)

    # sacred is on by default
    logger.setup_sacred(_run)

    # Run and train
    run_sequential(args=args, saia_args=saia_args, logger=logger)

    # Clean up after finishing
    print("Exiting Main")

    print("Stopping all threads")
    for t in threading.enumerate():
        if t.name != "MainThread":
            print("Thread {} is alive! Is daemon: {}".format(t.name, t.daemon))
            t.join(timeout=1)
            print("Thread joined")

    print("Exiting script")

    # Making sure framework really exits
    os._exit(os.EX_OK)


def run_exploration(args, runner):
    correct_trajs_all = []
    i = 0
    while i < args.patch_budget:
        traj, e_r = runner.run_single_exploration()
        if e_r >= args.ep_reward_th:
            correct_trajs_all.append(traj)
            i += 1
            print(f"Have found patch traj: {i}")
    return correct_trajs_all


def evaluate_sequential(args, runner):

    avg_return_list = []
    for j in range(0, 3):
        ep_returns = []
        for i in range(args.test_nepisode):
            _, e_r = runner.run(test_mode=True)
            ep_returns.append(e_r)
            print(f"------Iter: {j+1}, game NO: {i+1}, game_episode_return: {e_r}------")

        if args.save_replay:
            runner.save_replay()

        print(f"End Info: {runner.get_stats()}")
        avg_return = sum(ep_returns)/args.test_nepisode
        print(f"AVG EP returns: {avg_return}")
        avg_return_list.append(avg_return)
    print(f"AVG EP returns: {avg_return_list}, AVG: {np.mean(avg_return_list)}")
    time.sleep(6)
    runner.close_env()


def run_sequential(args, saia_args, logger):

    # Init runner so we can get env info
    runner = saia_r_REGISTRY[args.runner](args=args, logger=logger)

    # Set up schemes and groups here
    env_info = runner.get_env_info()
    args.n_agents = env_info["n_agents"]
    args.n_actions = env_info["n_actions"]
    args.state_shape = env_info["state_shape"]
    args.accumulated_episodes = getattr(args, "accumulated_episodes", None)

    saia_args.n_agents = env_info["n_agents"]
    saia_args.n_actions = [args.n_agents + 1, env_info["obs_shape"]]
    saia_args.state_shape = env_info["state_shape"]
    saia_args.accumulated_episodes = getattr(args, "accumulated_episodes", None)
    saia_args.agent = args.agent_saia

    if getattr(args, 'agent_own_state_size', False):
        args.agent_own_state_size = get_agent_own_state_size(args.env_args)

    # Default/Base scheme
    scheme = {
        "state": {"vshape": env_info["state_shape"]},
        "obs": {"vshape": env_info["obs_shape"], "group": "agents"},
        "actions": {"vshape": (1,), "group": "agents", "dtype": th.long},
        "avail_actions": {"vshape": (env_info["n_actions"],), "group": "agents", "dtype": th.int},
        "probs": {"vshape": (env_info["n_actions"],), "group": "agents", "dtype": th.float},
        "reward": {"vshape": (1,)},
        "terminated": {"vshape": (1,), "dtype": th.uint8},
    }
    """My Code: saia scheme"""
    saia_scheme = {
        "state": {"vshape": env_info["state_shape"]},
        "obs": {"vshape": env_info["obs_shape"], "group": "agents"},
        "actions1": {"vshape": (1,), "group": "saia_agent", "dtype": th.long},
        "actions2": {"vshape": (env_info["obs_shape"],), "dtype": th.float},
        "avail_actions1": {"vshape": (env_info["n_agents"] + 1), "group": "saia_agent", "dtype": th.int},
        "actions1_probs": {"vshape": (env_info["n_agents"] + 1), "group": "saia_agent", "dtype": th.float},
        "reward": {"vshape": (1,)},
        "terminated": {"vshape": (1,), "dtype": th.uint8},
    }
    groups = {
        "agents": args.n_agents,
        "saia_agent": 1
    }
    preprocess = {
        "actions": ("actions_onehot", [OneHot(out_dim=args.n_actions)])
    }
    """My Code: saia preprocess"""
    saia_preprocess = {
        "actions1": ("actions1_onehot", [OneHot(out_dim=saia_args.n_actions[0])])
    }

    buffer = ReplayBuffer(scheme, groups, args.buffer_size, env_info["episode_limit"] + 1,
                          preprocess=preprocess,
                          device="cpu" if args.buffer_cpu_only else args.device)
    saia_buffer = ReplayBuffer(saia_scheme, groups, args.buffer_size, env_info["episode_limit"] + 1,
                                 preprocess=saia_preprocess,
                                 device="cpu" if args.buffer_cpu_only else args.device)
    # Setup multi-agent controller here
    mac = mac_REGISTRY[args.mac](buffer.scheme, groups, args)
    """My Code: Setup saia agent controller here"""
    saia_mac = saia_mac_REGISTRY[saia_args.mac](saia_buffer.scheme, groups, saia_args)

    # Give runner the scheme
    runner.setup(scheme=scheme, groups=groups, preprocess=preprocess, mac=mac,
                 saia_mac=saia_mac, saia_preprocess=saia_preprocess, saia_scheme=saia_scheme)

    # Learner
    learner = le_REGISTRY[args.learner](mac, buffer.scheme, logger, args)
    """My Code: AIA Learner"""
    saia_args.learner = "saia_learner"
    saia_learner = saia_le_REGISTRY[saia_args.learner](saia_mac, saia_buffer.scheme, logger, saia_args)

    if saia_args.use_cuda:
        learner.cuda()
        """My Code"""
        saia_learner.cuda()

    # Load learner from checkpoint_path
    assert args.checkpoint_path != "", "Need checkpoint_path for agents_model."
    if not os.path.isdir(args.checkpoint_path):
        logger.console_logger.info(f"Checkpoint directory {args.checkpoint_path} doesn't exist!")
        return
    else:
        runner.t_env = load_checkpoint(args.checkpoint_path, args.load_step, learner, logger)

    # Load saia learner from checkpoint_path
    if args.saia_checkpoint_path != "":
        if not os.path.isdir(args.saia_checkpoint_path):
            logger.console_logger.info(f"AIA checkpoint directory {args.saia_checkpoint_path} doesn't exist!")
            return
        else:
            runner.saia_t_env = load_checkpoint(args.saia_checkpoint_path, args.saia_load_step, saia_learner, logger)

    if args.evaluate or args.save_replay or args.run_mode == "Test":
        evaluate_sequential(args, runner)
        return

    # start training
    episode = 0
    last_test_T = -args.test_interval - 1
    last_log_T = 0
    model_save_time = 0

    start_time = time.time()
    last_time = start_time

    logger.console_logger.info("Beginning training for {} timesteps".format(args.t_max))

    updates = 0
    while runner.saia_t_env <= args.t_max:

        # Run for a whole episode at a time

        with th.no_grad():
            episode_batch, _ = runner.run(test_mode=False)
            # buffer.insert_episode_batch(episode_batch)
            saia_buffer.insert_episode_batch(episode_batch)

        if saia_buffer.can_sample(args.batch_size):
            next_episode = episode + args.batch_size_run
            if args.accumulated_episodes and next_episode % args.accumulated_episodes != 0:
                continue

            episode_sample = saia_buffer.sample(args.batch_size)

            # Truncate batch to only filled timesteps
            max_ep_t = episode_sample.max_t_filled()
            episode_sample = episode_sample[:, :max_ep_t]

            if episode_sample.device != args.device:
                episode_sample.to(args.device)

            # learner.train(episode_sample, runner.t_env, episode)
            saia_learner.train(episode_sample, runner.saia_t_env, episode, updates)
            updates += 1
            del episode_sample

        # Execute test runs once in a while
        n_test_runs = max(1, args.test_nepisode // runner.batch_size)
        if (runner.saia_t_env - last_test_T) / args.test_interval >= 1.0:

            logger.console_logger.info("saia_t_env: {} / {}".format(runner.saia_t_env, args.t_max))
            logger.console_logger.info("Estimated time left: {}. Time passed: {}".format(
                    time_left(last_time, last_test_T, runner.saia_t_env, args.t_max), time_str(time.time() - start_time)))
            last_time = time.time()

            last_test_T = runner.saia_t_env
            for _ in range(n_test_runs):
                runner.run(test_mode=True)

        if args.save_model and (runner.saia_t_env - model_save_time >= args.save_model_interval or model_save_time == 0):
            model_save_time = runner.saia_t_env
            save_path = os.path.join(args.local_results_path, "saia_models",
                                     f"{args.env_args['map_name']}_{args.unique_token}", str(runner.saia_t_env))
            os.makedirs(save_path, exist_ok=True)
            logger.console_logger.info("Saving models to {}".format(save_path))

            # saia_learner should handle saving/loading -- delegate actor save/load to mac,
            # use appropriate filenames to do critics, optimizer states
            saia_learner.save_models(save_path)

        episode += args.batch_size_run

        if (runner.saia_t_env - last_log_T) >= args.log_interval:
            logger.log_stat("episode", episode, runner.saia_t_env)
            logger.print_recent_stats()
            last_log_T = runner.saia_t_env

    runner.close_env()
    logger.console_logger.info("Finished Training")


def load_checkpoint(checkpoint_path, load_step, learner, logger):
    time_steps = []
    timestep_to_load = 0
    # Go through all files in args.checkpoint_path
    for name in os.listdir(checkpoint_path):
        full_name = os.path.join(checkpoint_path, name)
        # Check if they are dirs the names of which are numbers
        if os.path.isdir(full_name) and name.isdigit():
            time_steps.append(int(name))
    if load_step == 0:
        # choose the max timestep
        timestep_to_load = max(time_steps)
    else:
        # choose the timestep closest to load_step
        timestep_to_load = min(time_steps, key=lambda x: abs(x - load_step))
    model_path = os.path.join(checkpoint_path, str(timestep_to_load))
    logger.console_logger.info("Loading model from {}".format(model_path))
    learner.load_models(model_path)
    return timestep_to_load


def args_sanity_check(config, _log):
    # set CUDA flags
    # config["use_cuda"] = True # Use cuda whenever possible!
    if config["use_cuda"] and not th.cuda.is_available():
        config["use_cuda"] = False
        _log.warning("CUDA flag use_cuda was switched OFF automatically because no CUDA devices are available!")

    if config["test_nepisode"] < config["batch_size_run"]:
        config["test_nepisode"] = config["batch_size_run"]
    else:
        config["test_nepisode"] = (config["test_nepisode"] // config["batch_size_run"]) * config["batch_size_run"]

    return config
