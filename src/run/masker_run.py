import datetime
import os
import pprint
import time
import threading
import torch as th
from types import SimpleNamespace as SN
from utils.logging import Logger
from utils.timehelper import time_left, time_str
from os.path import dirname, abspath

from learners import REGISTRY as le_REGISTRY
from runners import MASKER_REGISTRY as masker_r_REGISTRY
from controllers import REGISTRY as mac_REGISTRY
from controllers import MASKER_REGISTRY as masker_mac_REGISTRY
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
    masker_args = SN(**_config)
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
    masker_args.unique_token = unique_token
    if args.use_tensorboard:
        tb_logs_direc = os.path.join(dirname(dirname(dirname(abspath(__file__)))), "results", "tb_logs")
        tb_exp_direc = os.path.join(tb_logs_direc, "{}").format(unique_token)
        logger.setup_tb(tb_exp_direc)

    # sacred is on by default
    logger.setup_sacred(_run)

    # Run and train
    run_sequential(args=args, masker_args=masker_args, logger=logger)

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


def evaluate_sequential(args, runner):
    for _ in range(args.test_nepisode):
        runner.run(test_mode=True)

    if args.save_replay:
        runner.save_replay()

    runner.close_env()


def run_sequential(args, masker_args, logger):
    # Init runner so we can get env info
    runner = masker_r_REGISTRY[args.runner](args=args, logger=logger)

    # Set up schemes and groups here
    env_info = runner.get_env_info()
    args.n_agents = env_info["n_agents"]
    args.n_actions = env_info["n_actions"]
    args.state_shape = env_info["state_shape"]
    args.accumulated_episodes = getattr(args, "accumulated_episodes", None)

    masker_args.n_agents = env_info["n_agents"]
    masker_args.n_actions = 2
    masker_args.state_shape = env_info["state_shape"]
    masker_args.accumulated_episodes = getattr(args, "accumulated_episodes", None)

    if getattr(args, 'agent_own_state_size', False):
        args.agent_own_state_size = get_agent_own_state_size(args.env_args, args.run_mode)
        masker_args.agent_own_state_size = get_agent_own_state_size(masker_args.env_args, masker_args.run_mode)

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
    """My Code: masker scheme"""
    masker_scheme = {
        "state": {"vshape": env_info["state_shape"]},
        "obs": {"vshape": env_info["obs_shape"], "group": "agents"},
        "actions": {"vshape": (1,), "group": "agents", "dtype": th.long},
        "avail_actions": {"vshape": (2), "group": "agents", "dtype": th.int},
        "probs": {"vshape": (env_info["n_actions"],), "group": "agents", "dtype": th.float},
        "reward": {"vshape": (1,)},
        "terminated": {"vshape": (1,), "dtype": th.uint8},
    }
    groups = {
        "agents": args.n_agents
    }
    preprocess = {
        "actions": ("actions_onehot", [OneHot(out_dim=args.n_actions)])
    }
    """My Code: masker preprocess"""
    masker_preprocess = {
        "actions": ("actions_onehot", [OneHot(out_dim=masker_args.n_actions)])
    }

    buffer = ReplayBuffer(scheme, groups, args.buffer_size, env_info["episode_limit"] + 1,
                          preprocess=preprocess,
                          device="cpu" if args.buffer_cpu_only else args.device)
    masker_buffer = ReplayBuffer(masker_scheme, groups, args.buffer_size, env_info["episode_limit"] + 1,
                                 preprocess=masker_preprocess,
                                 device="cpu" if args.buffer_cpu_only else args.device)
    # Setup multiagent controller here
    mac = mac_REGISTRY[args.mac](buffer.scheme, groups, args)
    """My Code: Setup masker agent controller here"""
    masker_mac = masker_mac_REGISTRY[masker_args.mac](masker_buffer.scheme, groups, masker_args)

    # Give runner the scheme
    runner.setup(scheme=scheme, groups=groups, preprocess=preprocess, mac=mac,
                 masker_mac=masker_mac, masker_preprocess=masker_preprocess, masker_scheme=masker_scheme)

    # Learner
    learner = le_REGISTRY[args.learner](mac, buffer.scheme, logger, args)
    """My Code: Masker Learner"""
    masker_learner = le_REGISTRY[masker_args.learner](masker_mac, masker_buffer.scheme, logger, masker_args)

    if masker_args.use_cuda:
        learner.cuda()
        """My Code"""
        masker_learner.cuda()

    # Load learner from checkpoint_path
    assert args.checkpoint_path != "", "Need checkpoint_path for agents_model."
    if not os.path.isdir(args.checkpoint_path):
        logger.console_logger.info(f"Checkpoint directory {args.checkpoint_path} doesn't exist!")
        return
    else:
        runner.t_env = load_checkpoint(args.checkpoint_path, args.load_step, learner, logger)

    # Load masker learner from checkpoint_path
    if args.masker_checkpoint_path != "":
        if not os.path.isdir(args.masker_checkpoint_path):
            logger.console_logger.info(f"Masker checkpoint directory {args.masker_checkpoint_path} doesn't exist!")
            return
        else:
            runner.masker_t_env = load_checkpoint(args.masker_checkpoint_path, args.masker_load_step, masker_learner, logger)

    if args.evaluate or args.save_replay:
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

    while runner.masker_t_env <= args.t_max:

        # Run for a whole episode at a time

        with th.no_grad():
            episode_batch = runner.run(test_mode=False)
            # buffer.insert_episode_batch(episode_batch)
            masker_buffer.insert_episode_batch(episode_batch)

        if masker_buffer.can_sample(args.batch_size):
            next_episode = episode + args.batch_size_run
            if args.accumulated_episodes and next_episode % args.accumulated_episodes != 0:
                continue

            episode_sample = masker_buffer.sample(args.batch_size)

            # Truncate batch to only filled timesteps
            max_ep_t = episode_sample.max_t_filled()
            episode_sample = episode_sample[:, :max_ep_t]

            if episode_sample.device != args.device:
                episode_sample.to(args.device)

            # learner.train(episode_sample, runner.t_env, episode)
            masker_learner.train(episode_sample, runner.masker_t_env, episode)
            del episode_sample

        # Execute test runs once in a while
        n_test_runs = max(1, args.test_nepisode // runner.batch_size)
        if (runner.masker_t_env - last_test_T) / args.test_interval >= 1.0:

            logger.console_logger.info("masker_t_env: {} / {}".format(runner.masker_t_env, args.t_max))
            logger.console_logger.info("Estimated time left: {}. Time passed: {}".format(
                    time_left(last_time, last_test_T, runner.masker_t_env, args.t_max), time_str(time.time() - start_time)))
            last_time = time.time()

            last_test_T = runner.masker_t_env
            for _ in range(n_test_runs):
                runner.run(test_mode=True)

        if args.save_model and (runner.masker_t_env - model_save_time >= args.save_model_interval or model_save_time == 0):
            model_save_time = runner.masker_t_env
            save_path = os.path.join(args.local_results_path, "masker_models",
                                     f"{args.env_args['map_name']}_{args.unique_token}", str(runner.masker_t_env))
            # save_path = os.path.join(args.local_results_path, "models", args.unique_token, str(runner.t_env))
            # "results/models/{}".format(unique_token)
            os.makedirs(save_path, exist_ok=True)
            logger.console_logger.info("Saving models to {}".format(save_path))

            # masker_learner should handle saving/loading -- delegate actor save/load to mac,
            # use appropriate filenames to do critics, optimizer states
            masker_learner.save_models(save_path)

        episode += args.batch_size_run

        if (runner.masker_t_env - last_log_T) >= args.log_interval:
            logger.log_stat("episode", episode, runner.masker_t_env)
            logger.print_recent_stats()
            last_log_T = runner.masker_t_env

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
