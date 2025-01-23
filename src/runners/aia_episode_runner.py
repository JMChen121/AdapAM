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
        self.aia_t_env = 0

        self.attack_num = 0
        self.agent_step_num = 0
        self.all_step_num = 0

    def setup(self, scheme, groups, preprocess, mac, aia_scheme, aia_preprocess, aia_mac):
        self.new_batch = partial(EpisodeBatch, scheme, groups, self.batch_size, self.episode_limit + 1,
                                 preprocess=preprocess, device=self.args.device)
        self.mac = mac

        """# My code"""
        self.new_aia_batch = partial(EpisodeBatch, aia_scheme, groups, self.batch_size, self.episode_limit + 1,
                                        preprocess=aia_preprocess, device=self.args.device)
        self.aia_mac = aia_mac

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
        self.aia_batch = self.new_aia_batch()
        self.env.reset()
        self.t = 0

    def run(self, test_mode=False):
        self.reset()

        terminated = False
        episode_return = 0
        self.mac.init_hidden(batch_size=self.batch_size)
        self.aia_mac.init_hidden(batch_size=self.batch_size)

        while not terminated:
            pre_transition_data = {
                "state": [self.env.get_state()],
                "avail_actions": [self.env.get_avail_actions()],
                "obs": [self.env.get_obs()]
            }

            self.batch.update(pre_transition_data, ts=self.t)
            pre_transition_data_aia = {
                "state": [self.env.get_state()],
                "avail_actions1": [self.env.get_avail_aia_actions()],
                "avail_actions2": [self.env.get_avail_actions()],
                "obs": [self.env.get_obs()]
            }
            self.aia_batch.update(pre_transition_data_aia, ts=self.t)

            # Pass the entire batch of experiences up till now to the agents
            # Receive the actions for each agent at this timestep in a batch of size 1
            origin_actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)
            masker_actions, attack_actions, agents_importance = self.aia_mac.select_actions(self.aia_batch, t_ep=self.t, t_env=self.aia_t_env, test_mode=test_mode)

            # Fix memory leak
            cpu_actions = origin_actions.to("cpu").numpy()
            cpu_masker_actions, cpu_attack_actions = masker_actions.to("cpu").numpy(), attack_actions.to("cpu").numpy()

            if self.args.run_mode == "Test":
                # random

                # print importance_out
                reshaped_avail_actions = self.aia_batch["avail_actions1"][:, self.t]
                agents_importance[reshaped_avail_actions == 0] = -1e5

                probs = torch.nn.functional.softmax(agents_importance, dim=-1).cpu().detach().numpy()
                importance = np.array([round(x, 4) for x in probs[0, :, 0]])
                agent_rank = np.argsort(importance)
                alive_agent_rank = [a for a in agent_rank if (reshaped_avail_actions[0, :, 0].cpu() != 0)[a]]
                max_importance = importance[alive_agent_rank[-1]]
                mean_importance = np.mean(importance[reshaped_avail_actions[0, :, 0].cpu() != 0])
                diff_importance = max_importance - mean_importance
                print(f"Game time step: {self.t}. Agent importance rank: {alive_agent_rank}. Importance: {importance}. Importance diff: {diff_importance}")

                self.agent_step_num += len(alive_agent_rank)
                if diff_importance >= self.args.imp_th:
                    '''AIA'''
                    final_actions = attack_important_agent(origin_actions, masker_actions, attack_actions)
                    '''Random Agent'''
                    # new_masker_actions = torch.zeros_like(masker_actions)
                    # for i in range(min(self.args.important_num, len(alive_agent_rank))):
                    #     new_masker_actions[0][random.choice(alive_agent_rank)] = 1
                    # final_actions = attack_important_agent(origin_actions, new_masker_actions, attack_actions)
                    '''Random Action'''
                    # target_ids = [alive_agent_rank[-self.args.important_num]]
                    # final_actions = torch.tensor([replace_action_by_id(origin_actions[0], target_ids, self.aia_batch["avail_actions2"][:, self.t][0])])
                    '''Random Agent + Random Action'''
                    # target_ids = []
                    # for i in range(min(self.args.important_num, len(alive_agent_rank))):
                    #     target_ids.append(random.choice(alive_agent_rank))
                    # final_actions = torch.tensor([replace_action_by_id(origin_actions[0], target_ids, self.aia_batch["avail_actions2"][:, self.t][0])])

                    # Statistics
                    self.attack_num += self.args.important_num
                else:
                    final_actions = origin_actions
            else:
                final_actions = attack_important_agent(origin_actions, masker_actions, attack_actions)

            # TODO: attack_reward = (origin_actions!=final_actions) - attack_num
            reward, terminated, env_info = self.env.step(final_actions[0].to("cpu").numpy())
            reward = -reward
            episode_return += reward

            post_transition_data = {
                "actions": cpu_actions,
                "reward": [(reward,)],
                "terminated": [(terminated != env_info.get("episode_limit", False),)],
            }
            post_transition_data_aia = {
                "actions1": cpu_masker_actions,
                "actions2": cpu_attack_actions,
                "reward": [(reward,)],
                "terminated": [(terminated != env_info.get("episode_limit", False),)],
            }
            self.batch.update(post_transition_data, ts=self.t)
            self.aia_batch.update(post_transition_data_aia, ts=self.t)

            self.t += 1
            self.all_step_num += 1

        last_data = {
            "state": [self.env.get_state()],
            "avail_actions": [self.env.get_avail_actions()],
            "obs": [self.env.get_obs()]
        }
        self.batch.update(last_data, ts=self.t)
        last_aia_data = {
            "state": [self.env.get_state()],
            "avail_actions1": [self.env.get_avail_aia_actions()],
            "avail_actions2": [self.env.get_avail_actions()],
            "obs": [self.env.get_obs()]
        }
        self.aia_batch.update(last_aia_data, ts=self.t)

        # Select actions in the last stored state
        actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)
        masker_actions, attack_actions, agents_importance = self.aia_mac.select_actions(self.aia_batch, t_ep=self.t,
                                                                     t_env=self.aia_t_env, test_mode=test_mode)
        # Fix memory leak
        cpu_actions = actions.to("cpu").numpy()
        cpu_masker_actions, cpu_attack_actions = masker_actions.to("cpu").numpy(), attack_actions.to("cpu").numpy()
        self.batch.update({"actions": cpu_actions}, ts=self.t)
        self.aia_batch.update({
                "actions1": cpu_masker_actions,
                "actions2": cpu_attack_actions,
        }, ts=self.t)
        
        cur_stats = self.test_stats if test_mode else self.train_stats
        cur_returns = self.test_returns if test_mode else self.train_returns
        log_prefix = "test_" if test_mode else ""
        cur_stats.update({k: cur_stats.get(k, 0) + env_info.get(k, 0) for k in set(cur_stats) | set(env_info)})
        cur_stats["n_episodes"] = 1 + cur_stats.get("n_episodes", 0)
        cur_stats["ep_length"] = self.t + cur_stats.get("ep_length", 0)

        if not test_mode:
            self.t_env += self.t
            self.aia_t_env += self.t

        cur_returns.append(episode_return)

        if test_mode and (len(self.test_returns) == self.args.test_nepisode):
            self._log(cur_returns, cur_stats, log_prefix)
        elif self.aia_t_env - self.log_train_stats_t >= self.args.runner_log_interval:
            self._log(cur_returns, cur_stats, log_prefix)
            if hasattr(self.aia_mac.action_selector, "epsilon"):
                self.logger.log_stat("epsilon", self.aia_mac.action_selector.epsilon, self.aia_t_env)
            self.log_train_stats_t = self.aia_t_env

        return self.aia_batch, episode_return

    def _log(self, returns, stats, prefix):
        self.logger.log_stat(prefix + "return_mean", np.mean(returns), self.t_env)
        self.logger.log_stat(prefix + "return_std", np.std(returns), self.t_env)
        returns.clear()

        for k, v in stats.items():
            if k != "n_episodes":
                self.logger.log_stat(prefix + k + "_mean" , v/stats["n_episodes"], self.t_env)
        stats.clear()
