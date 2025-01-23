import random
from copy import deepcopy
from os.path import dirname, abspath, join

from envs import REGISTRY as env_REGISTRY
from functools import partial
from components.episode_buffer import EpisodeBatch
import numpy as np
import torch
import heapq
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
    for agent_id, agent_action in enumerate(origin_actions.tolist()):
        if agent_id in target_ids:   # mask
            # avail_action = avail_actions[agent_id][:2]  # stop action
            avail_action = avail_actions[agent_id]    # all available actions
            indices_avail_actions = [i for i, value in enumerate(avail_action) if value == 1]
            new_actions.append(random.choice(indices_avail_actions))
        else:
            new_actions.append(agent_action)
    return new_actions


def get_max_value_id(all_actions_val, actions_id, num=1):
    actions_val = [all_actions_val[i][actions_id[i]] for i in range(len(actions_id))]
    top_n_indices = heapq.nlargest(num, range(len(actions_val)), actions_val.__getitem__)
    return np.array(top_n_indices)


def find_patch_actions(patch_pack, target_ob, target_agent_id, original_action, avail_action, diff_th=0.5):
    new_action = original_action
    print("Start to find patch actions")
    min_diff = diff_th
    for ep_id, ep_traj in enumerate(patch_pack):
        for step in range(len(ep_traj["ob"])):
            patch_ob, patch_action = ep_traj["ob"][step][target_agent_id], ep_traj["actions"][step][target_agent_id]
            ob_diff = np.sum(np.abs(patch_ob - target_ob))
            if ob_diff <= min_diff and avail_action[patch_action] == 1:
                new_action = patch_action
                min_diff = ob_diff

    print(f"Old action: {original_action}. New action: {new_action}. Ob diff: {min_diff}")
    return new_action


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

        self.attack_valid = 0
        self.attack_invalid = 0

        self.patch_valid = 0
        self.patch_invalid = 0

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

    def get_stats(self):
        stats = self.env.get_stats()
        if self.args.need_attack:
            stats["attack_valid"] = self.attack_valid
            stats["attack_invalid"] = self.attack_invalid
            stats["attack_rate"] = self.attack_valid / (self.attack_valid + self.attack_invalid + 1e-5)
        if self.args.need_patch:
            stats["patch_valid"] = self.patch_valid
            stats["patch_invalid"] = self.patch_invalid
            stats["patch_rate"] = self.patch_valid / (self.patch_valid + self.patch_invalid + 1e-5)
        return stats

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

        target_agent_id = -1

        while not terminated:
            pre_transition_data = {
                "state": [self.env.get_state()],
                "avail_actions": [self.env.get_avail_actions()],
                "obs": deepcopy([self.env.get_obs()])
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

                # # 绘制可视化图-SMAC
                # agent_x_positions = [round(self.env.agents[i].pos.x, 2) for i in alive_agent_rank]
                # agent_y_positions = [round(self.env.agents[i].pos.y, 2) for i in alive_agent_rank]
                # plt.scatter(agent_x_positions, agent_y_positions)
                # for i in range(len(agent_x_positions)):
                #     plt.text(agent_x_positions[i], agent_y_positions[i], str(agent_rank[i]), fontsize=10, ha='left')
                # plt.title(f"Game time step: {self.t}")
                # plt.xlim(5, 25)  # range of x
                # plt.ylim(5, 25)  # range of y
                # plt.savefig(f"{self.args.file_obs_path}/Game-{self.env.battles_game}_Step-{self.t}.png")
                # plt.clf()
                # agent_rank_positions = [(round(self.env.agents[i].pos.x, 2), round(self.env.agents[i].pos.y, 2)) for i in alive_agent_rank]
                # print(f"Positions ranked by importance: {agent_rank_positions}")

                '''masker'''
                target_ids = agent_rank[:self.args.att_size]      # most important (min mask prob)
                # target_ids = agent_rank[-self.args.att_size:]     # most unimportant (max mask prob)
                '''Value Based'''
                # target_ids = get_max_value_id(self.mac.crt_agent_outs.reshape(self.masker_batch.batch_size * self.args.n_agents, -1).tolist(), origin_actions[0].tolist(), num=self.args.att_size)
                '''Randomly Choose'''
                # target_ids = np.random.choice(agent_rank, size=self.args.att_size)

                '''Attack: perturb observation'''
                if self.args.need_attack and target_ids.any() != -1:
                    origin_actions = self.get_attacked_actions(origin_actions, pre_transition_data, target_ids, test_mode, noise_w=self.args.noise_w)      #SMAC: 1     #GRF: 0.1

                '''not masking, use origin_actions'''
                final_actions = origin_actions[0]
                # final_actions = mask_actions(origin_actions[0], masker_actions[0], pre_transition_data["avail_actions"][0])
                '''replace to random action'''
                # final_actions = replace_action_by_id(origin_actions[0], target_ids, pre_transition_data["avail_actions"][0])
                mask_reward = 0
            else:
                final_actions = mask_actions(origin_actions[0], masker_actions[0], pre_transition_data["avail_actions"][0])
                mask_reward = np.sum(cpu_masker_actions == 1) * self.env.mask_reward

            reward, terminated, env_info = self.env.step(final_actions)
            print(f"Reward: {reward}")
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
        cur_stats.update({k: cur_stats.get(k, 0) + env_info.get(k, 0) for k in set(cur_stats) | set(env_info) if k != 'dumps'})
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

        return self.masker_batch, episode_return

    def get_attacked_actions(self, origin_actions, pre_transition_data, target_ids, test_mode, noise_w=1.0):
        for t_id in target_ids:
            attack_obs = pre_transition_data["obs"][0][t_id]
            # get noise range
            noise_range = (attack_obs.min() * noise_w, attack_obs.max() * noise_w)
            noise = np.random.uniform(noise_range[0], noise_range[1], attack_obs.shape[0])
            attack_obs += noise
        self.agent_batch.update(pre_transition_data, ts=self.t)
        # select actions of ob+noise
        attacked_actions = self.mac.select_actions(self.agent_batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode,
                                                   is_attack=True)
        if torch.equal(attacked_actions, origin_actions):
            self.attack_invalid += 1
        else:
            self.attack_valid += 1
        print(
            f"Attack valid: {torch.equal(attacked_actions, origin_actions) is False}. Original actions: {origin_actions[0].tolist()}. Attacked actions: {attacked_actions[0].tolist()}")
        return attacked_actions

    def _log(self, returns, stats, prefix):
        self.logger.log_stat(prefix + "return_mean", np.mean(returns), self.masker_t_env)
        self.logger.log_stat(prefix + "return_std", np.std(returns), self.masker_t_env)
        returns.clear()

        for k, v in stats.items():
            if k != "n_episodes":
                self.logger.log_stat(prefix + k + "_mean", v / stats["n_episodes"], self.masker_t_env)
        stats.clear()

    def run_single_exploration(self, test_mode=True):
        self.reset()

        terminated = False
        episode_return = 0
        self.mac.init_hidden(batch_size=self.batch_size)

        traj = {
            "state": [],
            "avail_actions": [],
            "ob": [],
            "actions": [],
            "reward": [],
        }

        while not terminated:
            pre_transition_data = {
                "state": [self.env.get_state()],
                "avail_actions": [self.env.get_avail_actions()],
                "obs": deepcopy([self.env.get_obs()])
            }
            self.agent_batch.update(pre_transition_data, ts=self.t)

            # Pass the entire batch of experiences up till now to the agents
            # Receive the actions for each agent at this timestep in a batch of size 1
            origin_actions = self.mac.select_actions(self.agent_batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)
            # Fix memory leak
            cpu_actions = origin_actions.to("cpu").numpy()

            '''not masking, use origin_actions'''
            final_actions = origin_actions[0]
            reward, terminated, env_info = self.env.step(final_actions)
            episode_return += reward

            post_transition_data = {
                "actions": cpu_actions,
                "reward": [(reward,)],
                "terminated": [(terminated != env_info.get("episode_limit", False),)],
            }
            self.agent_batch.update(post_transition_data, ts=self.t)

            self.t += 1

            traj["state"].append(pre_transition_data["state"][0])
            traj["avail_actions"].append(pre_transition_data["avail_actions"][0])
            traj["ob"].append(pre_transition_data["obs"][0])
            traj["actions"].append(post_transition_data["actions"][0])
            traj["reward"].append(post_transition_data["reward"][0])

        return traj, episode_return

    def run_with_patch(self, patch_pack, test_mode=True):
        self.reset()

        terminated = False
        episode_return = 0
        self.mac.init_hidden(batch_size=self.batch_size)
        self.masker_mac.init_hidden(batch_size=self.batch_size)

        while not terminated:
            pre_transition_data = {
                "state": [self.env.get_state()],
                "avail_actions": [self.env.get_avail_actions()],
                "obs": deepcopy([self.env.get_obs()])
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
            origin_actions = self.mac.select_actions(self.agent_batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)
            masker_actions = self.masker_mac.select_actions(self.masker_batch, t_ep=self.t, t_env=self.masker_t_env, test_mode=test_mode)
            # Fix memory leak
            cpu_actions = origin_actions.to("cpu").numpy()
            cpu_masker_actions = masker_actions.to("cpu").numpy()

            agent_outs = self.masker_mac.crt_agent_outs.reshape(self.masker_batch.batch_size * self.args.n_agents, -1)
            reshaped_avail_actions = self.masker_batch["avail_actions"][:, self.t].reshape(self.masker_batch.batch_size * self.args.n_agents, -1)
            agent_outs[reshaped_avail_actions == 0] = -1e5

            probs = torch.nn.functional.softmax(agent_outs, dim=-1).cpu().detach().numpy()
            agent_rank = np.argsort(probs[:, 1])

            target_id = -1
            '''masker'''
            # alive_agent_rank = agent_rank[probs[:, 1][agent_rank] != 0]
            # if len(alive_agent_rank) > 0:
            #     target_id = alive_agent_rank[0]     # most important (min mask prob)
            # target_id = alive_agent_rank[-1]     # most unimportant (max mask prob)
            '''Value Based'''
            # target_id = get_max_value_id(self.mac.crt_agent_outs.reshape(self.masker_batch.batch_size * self.args.n_agents, -1).tolist(), origin_actions[0].tolist())[0]
            '''Randomly Choose'''
            target_id = np.random.choice(agent_rank, size=1)[0]

            final_actions = origin_actions[0]

            '''search actions from patch package with similar observation'''
            if target_id != -1:
                patch_action_target = find_patch_actions(patch_pack, pre_transition_data["obs"][0][target_id], target_id, cpu_actions[0][target_id],
                                                         pre_transition_data["avail_actions"][0][target_id], diff_th=self.args.diff_th)
                if final_actions[target_id] == patch_action_target:
                    self.patch_invalid += 1
                else:
                    final_actions[target_id] = patch_action_target
                    self.patch_valid += 1

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

        return self.agent_batch, episode_return
