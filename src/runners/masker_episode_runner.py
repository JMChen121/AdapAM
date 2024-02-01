import random
from os.path import dirname, abspath, join

from envs import REGISTRY as env_REGISTRY
from functools import partial
from components.episode_buffer import EpisodeBatch
import numpy as np
import torch
import matplotlib.pyplot as plt


def mask_actions(origin_actions, mask, avail_actions):
    new_actions = []
    for agent_id, m in enumerate(mask):
        if m.item() == 1:   # mask
            indices_avail_actions = [i for i, value in enumerate(avail_actions[agent_id]) if value == 1]
            new_actions.append(random.choice(indices_avail_actions))
        else:
            new_actions.append(origin_actions[agent_id])
    return new_actions


def replace_action_by_id(origin_actions, target_ids, avail_actions):
    new_actions = []
    for agent_id, agent_action in enumerate(origin_actions):
        if agent_id in target_ids:   # mask
            avail_action = avail_actions[agent_id]
            indices_avail_actions = [i for i, value in enumerate(avail_action) if value == 1]
            new_actions.append(random.choice(indices_avail_actions))
        else:
            new_actions.append(agent_action)
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
        self.masker_t_env = 0
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
            masker_actions = self.masker_mac.select_actions(self.masker_batch, t_ep=self.t, t_env=self.masker_t_env,
                                                            test_mode=test_mode)
            # Fix memory leak
            cpu_actions = origin_actions.to("cpu").numpy()
            cpu_masker_actions = masker_actions.to("cpu").numpy()

            if self.args.run_mode == "Test":
                agent_outs = self.masker_mac.crt_agent_outs.reshape(self.masker_batch.batch_size * self.args.n_agents, -1)
                reshaped_avail_actions = self.masker_batch["avail_actions"][:, self.t].reshape(self.masker_batch.batch_size * self.args.n_agents, -1)
                agent_outs[reshaped_avail_actions == 0] = -1e5
                probs = torch.nn.functional.softmax(agent_outs, dim=-1).cpu().detach().numpy()
                mask_probs = [round(x, 4) for x in probs[:, 1]]
                agent_rank = np.argsort(probs[:, 1])
                alive_agent_rank = agent_rank[probs[:, 1][agent_rank] != 0]
                print(f"Game time step: {self.t}. Masker result: {cpu_masker_actions[0]}. Agent importance rank: {alive_agent_rank}. prob: {mask_probs}")

                # agent_x_positions = [round(self.env.agents[i].pos.x, 2) for i in alive_agent_rank]
                # agent_y_positions = [round(self.env.agents[i].pos.y, 2) for i in alive_agent_rank]
                # plt.scatter(agent_x_positions, agent_y_positions)
                # for i in range(len(agent_x_positions)):
                #     plt.text(agent_x_positions[i], agent_y_positions[i], str(agent_rank[i]), fontsize=10, ha='left')
                # plt.title(f"Game time step: {self.t}")
                # plt.xlim(5, 25)  # 设置 x 轴的范围
                # plt.ylim(5, 25)  # 设置 y 轴的范围
                # plt.savefig(f"{self.args.file_obs_path}/Game-{self.env.battles_game}_Step-{self.t}.png")
                # plt.clf()
                # agent_rank_positions = [(round(self.env.agents[i].pos.x, 2), round(self.env.agents[i].pos.y, 2)) for i in alive_agent_rank]
                # print(f"Positions ranked by importance: {agent_rank_positions}")

                # final_actions = origin_actions[0]
                # final_actions = mask_actions(origin_actions[0], masker_actions[0], pre_transition_data["avail_actions"][0])
                # target_ids = [agent_rank[0]]    # max important (min mask prob)
                # target_ids = [agent_rank[-1]]
                target_ids = np.random.choice(agent_rank, size=1)
                final_actions = replace_action_by_id(origin_actions[0], target_ids, pre_transition_data["avail_actions"][0])
            else:
                final_actions = mask_actions(origin_actions[0], masker_actions[0], pre_transition_data["avail_actions"][0])

            mask_reward = np.sum(cpu_masker_actions == 1) * 0.01
            # mask_reward = 0
            reward, terminated, env_info = self.env.step(final_actions)
            reward += mask_reward
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
        masker_actions = self.masker_mac.select_actions(self.masker_batch, t_ep=self.t, t_env=self.masker_t_env,
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
            self.masker_t_env += self.t

        cur_returns.append(episode_return)

        if test_mode and (len(self.test_returns) == self.args.test_nepisode):
            self._log(cur_returns, cur_stats, log_prefix)
        elif self.masker_t_env - self.log_train_stats_t >= self.args.runner_log_interval:
            self._log(cur_returns, cur_stats, log_prefix)
            if hasattr(self.masker_mac.action_selector, "epsilon"):
                self.logger.log_stat("epsilon", self.masker_mac.action_selector.epsilon, self.masker_t_env)
            self.log_train_stats_t = self.masker_t_env

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
        self.logger.log_stat(prefix + "return_mean", np.mean(returns), self.masker_t_env)
        self.logger.log_stat(prefix + "return_std", np.std(returns), self.masker_t_env)
        returns.clear()

        for k, v in stats.items():
            if k != "n_episodes":
                self.logger.log_stat(prefix + k + "_mean", v / stats["n_episodes"], self.masker_t_env)
        stats.clear()
