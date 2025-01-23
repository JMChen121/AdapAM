import pickle
import random
from copy import deepcopy

import imageio
import numpy as np
import time
import torch

from runners.shared.base_runner_masker import RunnerMasker


def _t2n(x):
    return x.detach().cpu().numpy()


def mask_actions(origin_actions, masker_actions):
    for thread_i in range(origin_actions.shape[0]):
        for agent_i in range(origin_actions.shape[1]):
            if masker_actions[thread_i, agent_i, 0] == 1:
                random_action_id = np.random.randint(origin_actions.shape[2])  # randomly selected action index
                origin_actions[thread_i, agent_i] = 0  # set origin actions to 0
                origin_actions[
                    thread_i, agent_i, random_action_id] = 1  # set origin actions to 1, at selected action index

    return origin_actions


def agent_action2random(origin_actions, target_agent_ids):
    for thread_i in range(origin_actions.shape[0]):
        for agent_id in target_agent_ids:
            random_action_id = np.random.randint(origin_actions.shape[2])  # randomly selected action index
            origin_actions[thread_i, agent_id] = 0  # set origin actions to 0
            origin_actions[
                thread_i, agent_id, random_action_id] = 1  # set origin actions to 1, at selected action index

    return origin_actions


def perturb_observation(share_ob, agent_ob, noise_w=1):
    share_ob_noise_range = (share_ob.min() * noise_w, share_ob.max() * noise_w)
    agent_ob_noise_range = (agent_ob.min() * noise_w, agent_ob.max() * noise_w)
    share_ob_noise = np.random.uniform(share_ob_noise_range[0], share_ob_noise_range[1], share_ob.shape)
    agent_ob_noise = np.random.uniform(agent_ob_noise_range[0], agent_ob_noise_range[1], agent_ob.shape)
    share_ob += share_ob_noise
    agent_ob += agent_ob_noise


class MPERunnerMasker(RunnerMasker):
    """Runner class to perform training, evaluation. and data collection for the MPEs. See parent class for details."""

    def __init__(self, config):
        super(MPERunnerMasker, self).__init__(config)

        self.attack_valid = 0
        self.attack_invalid = 0

        self.patch_trajs = None
        self.patch_valid = 0
        self.patch_invalid = 0

    def run(self):
        self.warmup()

        start = time.time()
        episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads

        for episode in range(episodes):
            if self.use_linear_lr_decay:
                self.trainer_masker.policy.lr_decay(episode, episodes)

            for step in range(self.episode_length):
                # Sample actions
                values, actions, action_log_probs, rnn_states, rnn_states_critic, actions_env, \
                    origin_values, origin_actions, origin_action_log_probs, origin_rnn_states, origin_rnn_states_critic = self.collect(
                        step)

                # Obser reward and next obs
                obs, rewards, dones, infos = self.envs.step(actions_env)

                data = obs, rewards, dones, infos, values, actions, action_log_probs, rnn_states, rnn_states_critic, \
                    origin_values, origin_actions, origin_action_log_probs, origin_rnn_states, origin_rnn_states_critic

                # insert data into buffer
                self.insert(data)

            # compute return and update network
            self.compute()
            train_infos = self.train()

            # post process
            total_num_steps = (episode + 1) * self.episode_length * self.n_rollout_threads

            # save model
            if episode % self.save_interval == 0 or episode == episodes - 1:
                self.save()

            # log information
            if episode % self.log_interval == 0:
                end = time.time()
                print("\n Scenario {} Algo {} Exp {} updates {}/{} episodes, total num timesteps {}/{}, FPS {}.\n"
                      .format(self.all_args.scenario_name,
                              self.algorithm_name,
                              self.experiment_name,
                              episode,
                              episodes,
                              total_num_steps,
                              self.num_env_steps,
                              int(total_num_steps / (end - start))))

                if self.env_name == "MPE":
                    env_infos = {}
                    for agent_id in range(self.num_agents):
                        idv_rews = []
                        for info in infos:
                            if 'individual_reward' in info[agent_id].keys():
                                idv_rews.append(info[agent_id]['individual_reward'])
                        agent_k = 'agent%i/individual_rewards' % agent_id
                        env_infos[agent_k] = idv_rews

                train_infos["average_episode_rewards"] = np.mean(self.buffer_masker.rewards) * self.episode_length
                print("average episode rewards is {}".format(train_infos["average_episode_rewards"]))
                self.log_train(train_infos, total_num_steps)
                self.log_env(env_infos, total_num_steps)

            # eval
            if episode % self.eval_interval == 0 and self.use_eval:
                self.eval(total_num_steps)

    def warmup(self):
        # reset env
        obs = self.envs.reset()

        # replay buffer
        if self.use_centralized_V:
            share_obs = obs.reshape(self.n_rollout_threads, -1)
            share_obs = np.expand_dims(share_obs, 1).repeat(self.num_agents, axis=1)
        else:
            share_obs = obs

        self.buffer.share_obs[0] = share_obs.copy()
        self.buffer.obs[0] = obs.copy()
        self.buffer_masker.share_obs[0] = share_obs.copy()
        self.buffer_masker.obs[0] = obs.copy()

    @torch.no_grad()
    def collect(self, step):
        self.trainer_masker.prep_rollout()
        origin_value, origin_action, origin_action_log_prob, origin_rnn_states, origin_rnn_states_critic \
            = self.trainer.policy.get_actions(np.concatenate(self.buffer.share_obs[step]),
                                              np.concatenate(self.buffer.obs[step]),
                                              np.concatenate(self.buffer.rnn_states[step]),
                                              np.concatenate(self.buffer.rnn_states_critic[step]),
                                              np.concatenate(self.buffer.masks[step]))
        origin_values = np.array(np.split(_t2n(origin_value), self.n_rollout_threads))
        origin_actions = np.array(np.split(_t2n(origin_action), self.n_rollout_threads))
        origin_action_log_probs = np.array(np.split(_t2n(origin_action_log_prob), self.n_rollout_threads))
        origin_rnn_states = np.array(np.split(_t2n(origin_rnn_states), self.n_rollout_threads))
        origin_rnn_states_critic = np.array(np.split(_t2n(origin_rnn_states_critic), self.n_rollout_threads))

        value, action, action_log_prob, rnn_states, rnn_states_critic \
            = self.trainer_masker.policy.get_actions(np.concatenate(self.buffer_masker.share_obs[step]),
                                                     np.concatenate(self.buffer_masker.obs[step]),
                                                     np.concatenate(self.buffer_masker.rnn_states[step]),
                                                     np.concatenate(self.buffer_masker.rnn_states_critic[step]),
                                                     np.concatenate(self.buffer_masker.masks[step]))

        # [self.envs, agents, dim]
        values = np.array(np.split(_t2n(value), self.n_rollout_threads))
        actions = np.array(np.split(_t2n(action), self.n_rollout_threads))
        action_log_probs = np.array(np.split(_t2n(action_log_prob), self.n_rollout_threads))
        rnn_states = np.array(np.split(_t2n(rnn_states), self.n_rollout_threads))
        rnn_states_critic = np.array(np.split(_t2n(rnn_states_critic), self.n_rollout_threads))
        # rearrange action
        if self.envs.action_space[0].__class__.__name__ == 'Discrete':
            actions_env = np.squeeze(np.eye(self.envs.action_space[0].n)[origin_actions], 2)
            actions_env = mask_actions(actions_env, actions)  # mask origin actions by mask actions
        elif self.envs.action_space[0].__class__.__name__ == 'MultiDiscrete':
            for i in range(self.envs.action_space[0].shape):
                uc_actions_env = np.eye(self.envs.action_space[0].high[i] + 1)[origin_actions[:, :, i]]
                uc_actions_env = mask_actions(uc_actions_env, actions)  # mask origin actions by mask actions
                if i == 0:
                    actions_env = uc_actions_env
                else:
                    actions_env = np.concatenate((actions_env, uc_actions_env), axis=2)
        else:
            raise NotImplementedError

        return values, actions, action_log_probs, rnn_states, rnn_states_critic, actions_env, \
            origin_values, origin_actions, origin_action_log_probs, origin_rnn_states, origin_rnn_states_critic

    def insert(self, data):
        obs, rewards, dones, infos, values, actions, action_log_probs, rnn_states, rnn_states_critic, \
            origin_values, origin_actions, origin_action_log_probs, origin_rnn_states, origin_rnn_states_critic = data

        origin_rnn_states[dones == True] = np.zeros(((dones == True).sum(), self.recurrent_N, self.hidden_size),
                                                    dtype=np.float32)
        origin_rnn_states_critic[dones == True] = np.zeros(
                ((dones == True).sum(), *self.buffer.rnn_states_critic.shape[3:]), dtype=np.float32)
        rnn_states[dones == True] = np.zeros(((dones == True).sum(), self.recurrent_N, self.hidden_size),
                                             dtype=np.float32)
        rnn_states_critic[dones == True] = np.zeros(
                ((dones == True).sum(), *self.buffer_masker.rnn_states_critic.shape[3:]), dtype=np.float32)
        masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)

        if self.use_centralized_V:
            share_obs = obs.reshape(self.n_rollout_threads, -1)
            share_obs = np.expand_dims(share_obs, 1).repeat(self.num_agents, axis=1)
        else:
            share_obs = obs

        self.buffer.insert(share_obs, obs, origin_rnn_states, origin_rnn_states_critic, origin_actions,
                           origin_action_log_probs, origin_values, rewards, masks)
        self.buffer_masker.insert(share_obs, obs, rnn_states, rnn_states_critic, actions, action_log_probs, values,
                                  rewards, masks)

    @torch.no_grad()
    def eval(self, total_num_steps):
        eval_episode_rewards = []
        eval_obs = self.eval_envs.reset()

        eval_rnn_states = np.zeros((self.n_eval_rollout_threads, *self.buffer_masker.rnn_states.shape[2:]),
                                   dtype=np.float32)
        eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)

        for eval_step in range(self.episode_length):
            self.trainer_masker.prep_rollout()
            eval_action, eval_rnn_states = self.trainer_masker.policy.act(np.concatenate(eval_obs),
                                                                          np.concatenate(eval_rnn_states),
                                                                          np.concatenate(eval_masks),
                                                                          deterministic=True)
            eval_actions = np.array(np.split(_t2n(eval_action), self.n_eval_rollout_threads))
            eval_rnn_states = np.array(np.split(_t2n(eval_rnn_states), self.n_eval_rollout_threads))

            if self.eval_envs.action_space[0].__class__.__name__ == 'MultiDiscrete':
                for i in range(self.eval_envs.action_space[0].shape):
                    eval_uc_actions_env = np.eye(self.eval_envs.action_space[0].high[i] + 1)[eval_actions[:, :, i]]
                    if i == 0:
                        eval_actions_env = eval_uc_actions_env
                    else:
                        eval_actions_env = np.concatenate((eval_actions_env, eval_uc_actions_env), axis=2)
            elif self.eval_envs.action_space[0].__class__.__name__ == 'Discrete':
                eval_actions_env = np.squeeze(np.eye(self.eval_envs.action_space[0].n)[eval_actions], 2)
            else:
                raise NotImplementedError

            # Obser reward and next obs
            eval_obs, eval_rewards, eval_dones, eval_infos = self.eval_envs.step(eval_actions_env)
            eval_episode_rewards.append(eval_rewards)

            eval_rnn_states[eval_dones == True] = np.zeros(
                    ((eval_dones == True).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32)
            eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)
            eval_masks[eval_dones == True] = np.zeros(((eval_dones == True).sum(), 1), dtype=np.float32)

        eval_episode_rewards = np.array(eval_episode_rewards)
        eval_env_infos = {}
        eval_env_infos['eval_average_episode_rewards'] = np.sum(np.array(eval_episode_rewards), axis=0)
        eval_average_episode_rewards = np.mean(eval_env_infos['eval_average_episode_rewards'])
        print("eval average episode rewards of agent: " + str(eval_average_episode_rewards))
        self.log_env(eval_env_infos, total_num_steps)

    @torch.no_grad()
    def render(self):
        """Visualize the env."""
        envs = self.envs

        all_frames = []
        all_episode_reward = []
        self.trainer.prep_rollout()
        self.trainer_masker.prep_rollout()

        self.attack_valid = 0
        self.attack_invalid = 0

        self.patch_valid = 0
        self.patch_invalid = 0
        for episode in range(self.all_args.render_episodes):
            obs = envs.reset()
            if self.use_centralized_V:
                share_obs = obs.reshape(self.n_rollout_threads, -1)
                share_obs = np.expand_dims(share_obs, 1).repeat(self.num_agents, axis=1)
            else:
                share_obs = obs

            if self.all_args.save_gifs:
                image = envs.render('rgb_array')[0][0]
                all_frames.append(image)
            elif self.all_args.use_render:
                envs.render('human')

            origin_rnn_states = np.zeros((self.n_rollout_threads, self.num_agents, self.recurrent_N, self.hidden_size),
                                         dtype=np.float32)
            origin_rnn_states_critic = np.zeros_like(origin_rnn_states)
            rnn_states = np.zeros((self.n_rollout_threads, self.num_agents, self.recurrent_N, self.hidden_size),
                                  dtype=np.float32)
            rnn_states_critic = np.zeros_like(rnn_states)
            masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)

            step_rewards = []

            for step in range(self.episode_length):
                calc_start = time.time()

                origin_value, origin_action, origin_action_logits, origin_next_rnn_states, origin_next_rnn_states_critic \
                    = self.trainer.policy.get_actions(np.concatenate(share_obs),
                                                      np.concatenate(obs),
                                                      np.concatenate(origin_rnn_states),
                                                      np.concatenate(origin_rnn_states_critic),
                                                      np.concatenate(masks))
                origin_actions = np.array(np.split(_t2n(origin_action), self.n_rollout_threads))
                origin_values = np.array(np.split(_t2n(origin_value), self.n_rollout_threads))[:, :, 0]
                origin_action_logits = np.array(np.split(_t2n(origin_action_logits), self.n_rollout_threads))
                origin_next_rnn_states = np.array(np.split(_t2n(origin_next_rnn_states), self.n_rollout_threads))
                origin_next_rnn_states_critic = np.array(
                    np.split(_t2n(origin_next_rnn_states_critic), self.n_rollout_threads))
                origin_agent_logits = np.sum(origin_action_logits, axis=2, keepdims=True)

                '''masker'''
                value, action, action_logit, rnn_states, rnn_states_critic \
                    = self.trainer_masker.policy.get_actions(np.concatenate(share_obs),
                                                             np.concatenate(obs),
                                                             np.concatenate(rnn_states),
                                                             np.concatenate(rnn_states_critic),
                                                             np.concatenate(masks),
                                                             deterministic=True,
                                                             need_logits=True)
                actions = np.array(np.split(_t2n(action), self.n_rollout_threads))
                values = np.array(np.split(_t2n(value), self.n_rollout_threads))
                action_logits = np.array(np.split(_t2n(action_logit.probs), self.n_rollout_threads))[:, :, 0]
                rnn_states = np.array(np.split(_t2n(rnn_states), self.n_rollout_threads))
                rnn_states_critic = np.array(np.split(_t2n(rnn_states_critic), self.n_rollout_threads))

                target_agent_ids = []
                for thread_id in range(self.n_rollout_threads):
                    target_agent_id = []
                    # --select most important agent by mask prob (min mask prob)
                    if self.all_args.agent_selection == "masker":
                        target_agent_id = np.argmin(action_logits[thread_id])
                        # target_agent_id = np.argpartition(action_logits[thread_id], 2)[:1]

                    '''Perturb Observation for Launching Attack'''
                    if self.all_args.need_attack is True:
                        perturb_observation(share_obs[thread_id][target_agent_id], obs[thread_id][target_agent_id])

                    target_agent_ids.append(target_agent_id)

                '''Launch Attack'''
                if self.all_args.need_attack is True:
                    origin_actions = self.get_attacked_actions(origin_actions, target_agent_ids, share_obs, obs,
                                                               origin_rnn_states, origin_rnn_states_critic, masks)
                '''Correct current action from patch trajs'''
                if self.all_args.need_patch is True:
                    origin_actions = self.get_patch_actions(origin_actions, target_agent_ids, obs)

                actions_env = None
                if self.envs.action_space[0].__class__.__name__ == 'Discrete':
                    actions_env = np.squeeze(np.eye(envs.action_space[0].n)[origin_actions], 2)
                    '''For RRD Calculation'''
                    if self.all_args.need_RRD:
                        actions_env = agent_action2random(actions_env, target_agent_ids)
                elif self.envs.action_space[0].__class__.__name__ == 'MultiDiscrete':
                    for i in range(self.envs.action_space[0].shape):
                        uc_actions_env = np.eye(envs.action_space[0].high[i] + 1)[origin_actions[:, :, i]]
                        '''For RRD Calculation'''
                        if self.all_args.need_RRD:
                            uc_actions_env = agent_action2random(uc_actions_env, target_agent_ids)
                        if i == 0:
                            actions_env = uc_actions_env
                        else:
                            actions_env = np.concatenate((actions_env, uc_actions_env), axis=2)
                else:
                    raise NotImplementedError

                # Obser reward and next obs
                obs, rewards, dones, infos = envs.step(actions_env)
                step_rewards.append(rewards)

                origin_next_rnn_states[dones == True] = np.zeros(
                        ((dones == True).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32)
                origin_next_rnn_states_critic[dones == True] = np.zeros(
                        ((dones == True).sum(), *self.buffer.rnn_states_critic.shape[3:]), dtype=np.float32)
                origin_rnn_states, origin_rnn_states_critic = origin_next_rnn_states, origin_next_rnn_states_critic
                rnn_states[dones == True] = np.zeros(
                        ((dones == True).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32)
                rnn_states_critic[dones == True] = np.zeros(
                        ((dones == True).sum(), *self.buffer_masker.rnn_states_critic.shape[3:]), dtype=np.float32)
                masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
                masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)

                if self.all_args.save_gifs:
                    image = envs.render('rgb_array')[0][0]
                    all_frames.append(image)
                    calc_end = time.time()
                    elapsed = calc_end - calc_start
                    if elapsed < self.all_args.ifi:
                        time.sleep(self.all_args.ifi - elapsed)
                elif self.all_args.use_render:
                    envs.render('human')

            episode_reward = np.mean(np.sum(np.array(step_rewards), axis=0))
            all_episode_reward.append(episode_reward)
            mean_episode_reward = np.mean(all_episode_reward)
            print(f"Episode {episode + 1} rewards is: {episode_reward}. Mean episode reward: {mean_episode_reward}")

        if self.all_args.save_gifs:
            imageio.mimsave(str(self.gif_dir) + '/render.gif', all_frames, duration=self.all_args.ifi)

        return mean_episode_reward

    def get_attacked_actions(self, origin_actions, agent_ids, share_obs, obs, rnn_states, rnn_states_critic, masks):

        _, new_action, _, _, _ \
            = self.trainer.policy.get_actions(np.concatenate(share_obs),
                                              np.concatenate(obs),
                                              np.concatenate(rnn_states),
                                              np.concatenate(rnn_states_critic),
                                              np.concatenate(masks))
        new_action = np.array(np.split(_t2n(new_action), self.n_rollout_threads))
        for thread_id in range(self.n_rollout_threads):
            agent = agent_ids[thread_id]
            if np.array_equal(new_action[thread_id][agent], origin_actions[thread_id][agent]):
                self.attack_invalid += 1
            else:
                self.attack_valid += 1
                origin_actions[thread_id][agent] = new_action[thread_id][agent]
        return origin_actions

    @torch.no_grad()
    def run_single_exploration(self):
        """Visualize the env."""
        envs = self.envs

        patch_ep = 0
        patch_trajs_all = []
        while patch_ep < self.all_args.patch_budget:
            obs = envs.reset()
            traj = {
                "obs": [],
                "rnn_states": [],
                "actions": [],
                "reward": [],
            }

            rnn_states = np.zeros((self.n_rollout_threads, self.num_agents, self.recurrent_N, self.hidden_size),
                                  dtype=np.float32)
            masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)

            step_rewards = []
            for step in range(self.episode_length):

                self.trainer.prep_rollout()
                traj["obs"].append(obs)
                traj["rnn_states"].append(rnn_states)
                action, rnn_states = self.trainer.policy.act(np.concatenate(obs),
                                                             np.concatenate(rnn_states),
                                                             np.concatenate(masks),
                                                             deterministic=True)
                actions = np.array(np.split(_t2n(action), self.n_rollout_threads))
                rnn_states = np.array(np.split(_t2n(rnn_states), self.n_rollout_threads))
                traj["actions"].append(actions)
                traj["rnn_states"].append(rnn_states)

                if envs.action_space[0].__class__.__name__ == 'MultiDiscrete':
                    for i in range(envs.action_space[0].shape):
                        uc_actions_env = np.eye(envs.action_space[0].high[i] + 1)[actions[:, :, i]]
                        if i == 0:
                            actions_env = uc_actions_env
                        else:
                            actions_env = np.concatenate((actions_env, uc_actions_env), axis=2)
                elif envs.action_space[0].__class__.__name__ == 'Discrete':
                    actions_env = np.squeeze(np.eye(envs.action_space[0].n)[actions], 2)
                else:
                    raise NotImplementedError

                # Obser reward and next obs
                obs, rewards, dones, infos = envs.step(actions_env)
                step_rewards.append(rewards)

                rnn_states[dones == True] = np.zeros(((dones == True).sum(), self.recurrent_N, self.hidden_size),
                                                     dtype=np.float32)
                masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
                masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)

            episode_reward = np.mean(np.sum(np.array(step_rewards), axis=0))

            if episode_reward > self.all_args.ep_reward_th:
                patch_ep += 1
                patch_trajs_all.append(traj)
                print(f"Episode {patch_ep} rewards is: {episode_reward}, added to patch.")

        patch_path = f"{self.all_args.model_dir}/patch_trajs.pkl"
        with open(patch_path, 'wb') as f:
            pickle.dump(patch_trajs_all, f)

    def load_patch(self):
        patch_path = f"{self.all_args.model_dir}/patch_trajs.pkl"
        with open(patch_path, 'rb') as f:
            self.patch_trajs = pickle.load(f)

    def get_patch_actions(self, origin_actions, agent_ids, obs):
        for thread_id in range(self.n_rollout_threads):
            agent = agent_ids[thread_id]
            ob = obs[thread_id][agent]
            new_action = deepcopy(origin_actions[thread_id][agent])

            # search similar ob from patch
            min_diff = self.all_args.diff_th
            for patch_id, patch_traj in enumerate(self.patch_trajs):
                if patch_id >= self.all_args.patch_budget:
                    break
                for p_step in range(len(patch_traj["obs"])):
                    patch_step_ob = patch_traj["obs"][p_step]
                    patch_action_ob = patch_traj["actions"][p_step]
                    for p_thread_id in range(patch_step_ob.shape[0]):
                        for p_agent_id in range(patch_step_ob.shape[1]):
                            patch_ob, patch_action = patch_step_ob[p_thread_id][p_agent_id], patch_action_ob[p_thread_id][p_agent_id]

                            ob_diff = np.sum(np.abs(patch_ob - ob))
                            if ob_diff <= min_diff:
                                new_action = patch_action
                                min_diff = ob_diff

            if np.array_equal(new_action, origin_actions[thread_id][agent]):
                self.patch_invalid += 1
            else:
                self.patch_valid += 1
                origin_actions[thread_id][agent] = new_action
        return origin_actions

