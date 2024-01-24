import random

from envs import REGISTRY as env_REGISTRY
from functools import partial
from components.episode_buffer import EpisodeBatch
import numpy as np
import torch


def mask_actions(origin_actions, mask, avail_actions):
    new_actions = []
    for agent_id, m in enumerate(mask):
        if m.item() == 1:
            indices_avail_actions = [i for i, value in enumerate(avail_actions[agent_id]) if value == 1]
            new_actions.append(random.choice(indices_avail_actions))
        else:
            new_actions.append(origin_actions[agent_id])
    return new_actions


class EpisodeRunner:

    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        self.batch_size = self.args.batch_size_run
        assert self.batch_size == 1

        self.env = env_REGISTRY[self.args.env](**self.args.env_args, run_mode=self.args.run_mode)
        self.episode_limit = self.env.episode_limit
        self.t = 0

        self.t_env = 0

        self.train_returns = []
        self.test_returns = []
        self.train_stats = {}
        self.test_stats = {}

        # Log the first run
        self.log_train_stats_t = -1000000

        """My params"""
        self.agents_attack_times = []
        self.agents_survival_time = []

    def setup(self, scheme, groups, preprocess, mac, masker_scheme, masker_preprocess, masker_mac):
        self.new_batch = partial(EpisodeBatch, scheme, groups, self.batch_size, self.episode_limit + 1,
                                 preprocess=preprocess, device=self.args.device)
        self.mac = mac

        """# My code"""
        self.new_masker_batch = partial(EpisodeBatch, masker_scheme, groups, self.batch_size, self.episode_limit + 1,
                                        preprocess=masker_preprocess, device=self.args.device)
        self.masker_mac = masker_mac

    def get_env_info(self):
        return self.env.get_env_info()

    def save_replay(self):
        self.env.save_replay()

    def close_env(self):
        self.env.close()

    def reset(self):
        self.agent_batch = self.new_batch()
        self.masker_batch = self.new_masker_batch()
        self.env.reset()
        self.t = 0

    def run(self, test_mode=False):
        self.reset()
        # TODO: collect self.masker_batch

        terminated = False
        episode_return = 0
        self.mac.init_hidden(batch_size=self.batch_size)
        self.masker_mac.init_hidden(batch_size=self.batch_size)

        while not terminated:
            pre_transition_data = {
                "state": [self.env.get_state()],
                "avail_actions": [self.env.get_avail_actions()],
                "obs": [self.env.get_obs()]
            }
            self.agent_batch.update(pre_transition_data, ts=self.t)
            pre_transition_data_masker = {
                "state": [self.env.get_state()],
                "avail_actions": [self.env.get_avail_masker_actions()],
                "obs": [self.env.get_obs()]
            }
            self.masker_batch.update(pre_transition_data_masker, ts=self.t)

            # Pass the entire batch of experiences up till now to the agents
            # Receive the actions for each agent at this timestep in a batch of size 1
            origin_actions = self.mac.select_actions(self.agent_batch, t_ep=self.t, t_env=self.t_env,
                                                     test_mode=test_mode)
            masker_actions = self.masker_mac.select_actions(self.masker_batch, t_ep=self.t, t_env=self.t_env,
                                                            test_mode=test_mode)
            # Fix memory leak
            cpu_actions = origin_actions.to("cpu").numpy()
            cpu_masker_actions = masker_actions.to("cpu").numpy()

            final_actions = mask_actions(origin_actions[0], masker_actions[0], pre_transition_data["avail_actions"][0])
            reward, terminated, env_info = self.env.step(final_actions)
            episode_return += reward

            post_transition_data = {
                "actions": cpu_actions,
                "reward": [(reward,)],
                "terminated": [(terminated != env_info.get("episode_limit", False),)],
            }
            post_transition_data_masker = {
                "actions": cpu_masker_actions,
                "reward": [(reward,)],
                "terminated": [(terminated != env_info.get("episode_limit", False),)],
            }
            self.agent_batch.update(post_transition_data, ts=self.t)
            self.masker_batch.update(post_transition_data_masker, ts=self.t)

            self.t += 1

        last_data = {
            "state": [self.env.get_state()],
            "avail_actions": [self.env.get_avail_actions()],
            "obs": [self.env.get_obs()]
        }
        self.agent_batch.update(last_data, ts=self.t)
        last_masker_data = {
            "state": [self.env.get_state()],
            "avail_actions": [self.env.get_avail_masker_actions()],
            "obs": [self.env.get_obs()]
        }
        self.masker_batch.update(last_masker_data, ts=self.t)

        # Select actions in the last stored state
        origin_actions = self.mac.select_actions(self.agent_batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)
        masker_actions = self.masker_mac.select_actions(self.masker_batch, t_ep=self.t, t_env=self.t_env,
                                                        test_mode=test_mode)
        # Fix memory leak
        cpu_actions = origin_actions.to("cpu").numpy()
        cpu_masker_actions = masker_actions.to("cpu").numpy()
        self.agent_batch.update({"actions": cpu_actions}, ts=self.t)
        self.masker_batch.update({"actions": cpu_masker_actions}, ts=self.t)

        cur_stats = self.test_stats if test_mode else self.train_stats
        cur_returns = self.test_returns if test_mode else self.train_returns
        log_prefix = "test_" if test_mode else ""
        cur_stats.update({k: cur_stats.get(k, 0) + env_info.get(k, 0) for k in set(cur_stats) | set(env_info)})
        cur_stats["n_episodes"] = 1 + cur_stats.get("n_episodes", 0)
        cur_stats["ep_length"] = self.t + cur_stats.get("ep_length", 0)

        if not test_mode:
            self.t_env += self.t

        cur_returns.append(episode_return)

        if test_mode and (len(self.test_returns) == self.args.test_nepisode):
            self._log(cur_returns, cur_stats, log_prefix)
        elif self.t_env - self.log_train_stats_t >= self.args.runner_log_interval:
            self._log(cur_returns, cur_stats, log_prefix)
            if hasattr(self.mac.action_selector, "epsilon"):
                self.logger.log_stat("epsilon", self.mac.action_selector.epsilon, self.t_env)
            self.log_train_stats_t = self.t_env

        """My Statistics"""
        # agents_actions = self.agent_batch.data.transition_data['actions'][0].T[0]
        #
        # for i in range(len(agents_actions)):
        #     # for attack times
        #     attack_times = torch.sum(agents_actions[i][:] > int(self.env.n_actions_no_attack)).item()
        #     # for survival time
        #     survival_time = (agents_actions[i] != 0).nonzero().shape[0]
        #     if len(self.agents_attack_times) == 0:
        #         self.agents_attack_times = [0] * len(agents_actions)
        #         self.agents_survival_time = [0] * len(agents_actions)
        #     else:
        #         self.agents_attack_times[i] += attack_times
        #         self.agents_survival_time[i] += survival_time

        """My output info for test, about selecting agent by max/min attack and survival"""
        # max_attacks, min_attacks = max(self.agents_attack_times), min(self.agents_attack_times)
        # max_a_index, min_a_index = self.agents_attack_times.index(max_attacks), self.agents_attack_times.index(min_attacks)
        # print(f"Max attacks--id(times) {max_a_index}({max_attacks}); Min attacks--id(times) {min_a_index}({min_attacks})")
        # max_survivals, min_survivals = max(self.agents_survival_time), min(self.agents_survival_time)
        # max_s_index, min_s_index = self.agents_survival_time.index(max_survivals), self.agents_survival_time.index(min_survivals)
        # print(f"Max survivals--id(times) {max_s_index}({max_survivals}); Min survivals--id(times) {min_s_index}({min_survivals})")

        return self.masker_batch

    def _log(self, returns, stats, prefix):
        self.logger.log_stat(prefix + "return_mean", np.mean(returns), self.t_env)
        self.logger.log_stat(prefix + "return_std", np.std(returns), self.t_env)
        returns.clear()

        for k, v in stats.items():
            if k != "n_episodes":
                self.logger.log_stat(prefix + k + "_mean", v / stats["n_episodes"], self.t_env)
        stats.clear()
