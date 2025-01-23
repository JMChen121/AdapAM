import random

from envs import REGISTRY as env_REGISTRY
from functools import partial
from components.episode_buffer import EpisodeBatch
import numpy as np
import torch

from runners.masker_episode_runner import replace_action_by_id


def attack_important_agent(origin_actions, masker_actions, attack_actions):
    attacked_actions = origin_actions.clone()
    attack_indices = masker_actions == 1
    attacked_actions[attack_indices] = attack_actions[attack_indices]
    return attacked_actions

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
        self.saia_t_env = 0

        self.attack_num = 0
        self.agent_step_num = 0
        self.all_step_num = 0

    def setup(self, scheme, groups, preprocess, mac, saia_scheme, saia_preprocess, saia_mac):
        self.new_batch = partial(EpisodeBatch, scheme, groups, self.batch_size, self.episode_limit + 1,
                                 preprocess=preprocess, device=self.args.device)
        self.mac = mac

        """# My code"""
        self.new_saia_batch = partial(EpisodeBatch, saia_scheme, groups, self.batch_size, self.episode_limit + 1,
                                        preprocess=saia_preprocess, device=self.args.device)
        self.saia_mac = saia_mac

    def get_env_info(self):
        return self.env.get_env_info()

    def save_replay(self):
        self.env.save_replay()

    def get_stats(self):
        stats = self.env.get_stats()
        stats["attack_num"] = self.attack_num
        stats["agent_step_num"] = self.agent_step_num
        stats["attack_rate"] = self.attack_num / (self.agent_step_num + 1e-8)
        stats["attack_step_rate"] = self.attack_num / (self.all_step_num + 1e-8)
        return stats

    def close_env(self):
        self.env.close()

    def reset(self):
        self.batch = self.new_batch()
        self.saia_batch = self.new_saia_batch()
        self.env.reset()
        self.t = 0

    def run(self, test_mode=False):
        self.reset()

        terminated = False
        episode_return = 0
        self.mac.init_hidden(batch_size=self.batch_size)
        self.saia_mac.init_hidden(batch_size=self.batch_size)

        while not terminated:

            pre_transition_data_saia = {
                "state": [self.env.get_state()],
                "avail_actions1": [self.env.get_avail_saia_actions()],
                "obs": [self.env.get_obs()]
            }
            self.saia_batch.update(pre_transition_data_saia, ts=self.t)

            # Pass the entire batch of experiences up till now to the agents
            # Receive the actions for each agent at this timestep in a batch of size 1
            vic_agent, vic_agent_q, perturbation, log_prob, mean = self.saia_mac.select_actions(self.saia_batch, t_ep=self.t, t_env=self.saia_t_env, test_mode=test_mode)
            cpu_vic_agent, cpu_perturbation = vic_agent.to("cpu").numpy(), perturbation.detach().cpu().numpy()

            pre_transition_data = {
                "state": [self.env.get_state()],
                "avail_actions": [self.env.get_avail_actions()],
                "obs": [self.env.get_obs()]
            }
            # attack ob of victim using perturbation
            pre_transition_data["obs"][0][self.t][vic_agent[0][0]] = cpu_perturbation[0]
            self.batch.update(pre_transition_data, ts=self.t)
            origin_actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)

            # Fix memory leak
            cpu_actions = origin_actions.to("cpu").numpy()

            if self.args.run_mode == "Test":
                pass

            # TODO: attack_reward = (origin_actions!=final_actions) - attack_num
            reward, terminated, env_info = self.env.step(origin_actions[0].to("cpu").numpy())
            reward = -reward
            episode_return += reward

            post_transition_data = {
                "actions": cpu_actions,
                "reward": [(reward,)],
                "terminated": [(terminated != env_info.get("episode_limit", False),)],
            }
            post_transition_data_saia = {
                "actions1": cpu_vic_agent,
                "actions2": cpu_perturbation,
                "reward": [(reward,)],
                "terminated": [(terminated != env_info.get("episode_limit", False),)],
                "actions1_probs": vic_agent_q.unsqueeze(1).to("cpu"),
            }
            self.batch.update(post_transition_data, ts=self.t)
            self.saia_batch.update(post_transition_data_saia, ts=self.t)

            self.t += 1
            self.all_step_num += 1

        last_saia_data = {
            "state": [self.env.get_state()],
            "avail_actions1": [self.env.get_avail_saia_actions()],
            "obs": [self.env.get_obs()]
        }
        self.saia_batch.update(last_saia_data, ts=self.t)
        # Select actions in the last stored state
        vic_agent, vic_agent_q, perturbation, log_prob, mean = self.saia_mac.select_actions(self.saia_batch, t_ep=self.t, t_env=self.saia_t_env, test_mode=test_mode)
        cpu_vic_agent, cpu_perturbation = vic_agent.to("cpu").numpy(), perturbation.detach().cpu().numpy()

        last_data = {
            "state": [self.env.get_state()],
            "avail_actions": [self.env.get_avail_actions()],
            "obs": [self.env.get_obs()]
        }
        last_data["obs"][0][self.t][vic_agent[0][0]] = cpu_perturbation[0]
        self.batch.update(last_data, ts=self.t)
        actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)
        # Fix memory leak
        cpu_actions = actions.to("cpu").numpy()
        self.batch.update({"actions": cpu_actions}, ts=self.t)
        self.saia_batch.update({
                "actions1": cpu_vic_agent,
                "actions2": cpu_perturbation,
                "actions1_probs": vic_agent_q.unsqueeze(1).to("cpu"),
        }, ts=self.t)
        
        cur_stats = self.test_stats if test_mode else self.train_stats
        cur_returns = self.test_returns if test_mode else self.train_returns
        log_prefix = "test_" if test_mode else ""
        cur_stats.update({k: cur_stats.get(k, 0) + env_info.get(k, 0) for k in set(cur_stats) | set(env_info)})
        cur_stats["n_episodes"] = 1 + cur_stats.get("n_episodes", 0)
        cur_stats["ep_length"] = self.t + cur_stats.get("ep_length", 0)

        if not test_mode:
            self.t_env += self.t
            self.saia_t_env += self.t

        cur_returns.append(episode_return)

        if test_mode and (len(self.test_returns) == self.args.test_nepisode):
            self._log(cur_returns, cur_stats, log_prefix)
        elif self.saia_t_env - self.log_train_stats_t >= self.args.runner_log_interval:
            self._log(cur_returns, cur_stats, log_prefix)
            if hasattr(self.saia_mac.action_selector, "epsilon"):
                self.logger.log_stat("epsilon", self.saia_mac.action_selector.epsilon, self.saia_t_env)
            self.log_train_stats_t = self.saia_t_env

        return self.saia_batch, episode_return

    def _log(self, returns, stats, prefix):
        self.logger.log_stat(prefix + "return_mean", np.mean(returns), self.t_env)
        self.logger.log_stat(prefix + "return_std", np.std(returns), self.t_env)
        returns.clear()

        for k, v in stats.items():
            if k != "n_episodes":
                self.logger.log_stat(prefix + k + "_mean" , v/stats["n_episodes"], self.t_env)
        stats.clear()
