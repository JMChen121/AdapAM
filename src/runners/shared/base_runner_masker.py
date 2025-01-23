import wandb
import os
import numpy as np
import torch
from tensorboardX import SummaryWriter
from utils.shared_buffer import SharedReplayBuffer
from gym.spaces import Discrete

def _t2n(x):
    """Convert torch tensor to a numpy array."""
    return x.detach().cpu().numpy()


class RunnerMasker(object):
    """
    Base class for training recurrent policies.
    :param config: (dict) Config dictionary containing parameters for training.
    """

    def __init__(self, config):

        self.all_args = config['all_args']
        self.envs = config['envs']
        self.eval_envs = config['eval_envs']
        self.device = config['device']
        self.num_agents = config['num_agents']
        if config.__contains__("render_envs"):
            self.render_envs = config['render_envs']

            # parameters
        self.env_name = self.all_args.env_name
        self.algorithm_name = self.all_args.algorithm_name
        self.experiment_name = self.all_args.experiment_name
        self.use_centralized_V = self.all_args.use_centralized_V
        self.use_obs_instead_of_state = self.all_args.use_obs_instead_of_state
        self.num_env_steps = self.all_args.num_env_steps
        self.episode_length = self.all_args.episode_length
        self.n_rollout_threads = self.all_args.n_rollout_threads
        self.n_eval_rollout_threads = self.all_args.n_eval_rollout_threads
        self.n_render_rollout_threads = self.all_args.n_render_rollout_threads
        self.use_linear_lr_decay = self.all_args.use_linear_lr_decay
        self.hidden_size = self.all_args.hidden_size
        self.use_wandb = self.all_args.use_wandb
        self.use_render = self.all_args.use_render
        self.recurrent_N = self.all_args.recurrent_N

        # interval
        self.save_interval = self.all_args.save_interval
        self.use_eval = self.all_args.use_eval
        self.eval_interval = self.all_args.eval_interval
        self.log_interval = self.all_args.log_interval

        # dir
        self.model_dir = self.all_args.model_dir

        '''My params for masker'''
        self.run_policy = self.all_args.run_policy
        self.masker_reward = self.all_args.masker_reward

        if self.use_render:
            self.run_dir = config["run_dir"]
            self.gif_dir = str(self.run_dir / 'gifs')
            if not os.path.exists(self.gif_dir):
                os.makedirs(self.gif_dir)

        else:
            if self.use_wandb:
                self.save_dir = str(wandb.run.dir)
                self.run_dir = str(wandb.run.dir)
            else:
                self.run_dir = config["run_dir"]
                self.log_dir = str(self.run_dir / 'logs')
                if not os.path.exists(self.log_dir):
                    os.makedirs(self.log_dir)
                self.writter = SummaryWriter(self.log_dir)
                self.save_dir = str(self.run_dir / 'models')
                if not os.path.exists(self.save_dir):
                    os.makedirs(self.save_dir)

        if self.algorithm_name == "mat" or self.algorithm_name == "mat_dec":
            from mpe.algorithms.mat.mat_trainer import MATTrainer as TrainAlgo
            from mpe.algorithms.mat.algorithm.transformer_policy import TransformerPolicy as Policy
        else:
            from learners.r_mappo import R_MAPPO as TrainAlgo
            from learners.rMAPPOPolicy import R_MAPPOPolicy as Policy

        share_observation_space = self.envs.share_observation_space[0] if self.use_centralized_V else \
            self.envs.observation_space[0]

        print("obs_space: ", self.envs.observation_space)
        print("share_obs_space: ", self.envs.share_observation_space)
        masker_action_space = Discrete(2)
        print("act_space: ", masker_action_space)

        # policy network
        self.policy = Policy(self.all_args, self.envs.observation_space[0], share_observation_space,
                             self.envs.action_space[0], device=self.device)
        self.policy_masker = Policy(self.all_args, self.envs.observation_space[0], share_observation_space,
                                    masker_action_space, device=self.device)

        if self.model_dir is not None:
            self.restore(self.model_dir)
            # for masker
            self.policy.actor.eval()
            self.policy.critic.eval()

        # algorithm
        self.trainer = TrainAlgo(self.all_args, self.policy, device=self.device)
        self.trainer_masker = TrainAlgo(self.all_args, self.policy_masker, device=self.device)

        # buffer
        self.buffer = SharedReplayBuffer(self.all_args,
                                         self.num_agents,
                                         self.envs.observation_space[0],
                                         share_observation_space,
                                         self.envs.action_space[0])
        self.buffer_masker = SharedReplayBuffer(self.all_args,
                                                self.num_agents,
                                                self.envs.observation_space[0],
                                                share_observation_space,
                                                masker_action_space)

    def run(self):
        """Collect training data, perform training updates, and evaluate policy."""
        raise NotImplementedError

    def warmup(self):
        """Collect warmup pre-training data."""
        raise NotImplementedError

    def collect(self, step):
        """Collect rollouts for training."""
        raise NotImplementedError

    def insert(self, data):
        """
        Insert data into buffer.
        :param data: (Tuple) data to insert into training buffer.
        """
        raise NotImplementedError

    @torch.no_grad()
    def compute(self):
        """Calculate returns for the collected data."""
        self.trainer_masker.prep_rollout()
        if self.algorithm_name == "mat" or self.algorithm_name == "mat_dec":
            next_values = self.trainer_masker.policy.get_values(np.concatenate(self.buffer_masker.share_obs[-1]),
                                                         np.concatenate(self.buffer_masker.obs[-1]),
                                                         np.concatenate(self.buffer_masker.rnn_states_critic[-1]),
                                                         np.concatenate(self.buffer_masker.masks[-1]))
        else:
            next_values = self.trainer_masker.policy.get_values(np.concatenate(self.buffer_masker.share_obs[-1]),
                                                         np.concatenate(self.buffer_masker.rnn_states_critic[-1]),
                                                         np.concatenate(self.buffer_masker.masks[-1]))
        next_values = np.array(np.split(_t2n(next_values), self.n_rollout_threads))
        self.buffer_masker.compute_returns(next_values, self.trainer_masker.value_normalizer)

    def train(self):
        """Train policies with data in buffer. """
        self.trainer_masker.prep_training()
        train_infos = self.trainer_masker.train(self.buffer_masker)
        self.buffer_masker.after_update()
        return train_infos

    def save(self, episode=0):
        """Save policy's actor and critic networks."""
        policy_actor = self.trainer_masker.policy.actor
        torch.save(policy_actor.state_dict(), str(self.save_dir) + "/actor.pt")
        policy_critic = self.trainer_masker.policy.critic
        torch.save(policy_critic.state_dict(), str(self.save_dir) + "/critic.pt")

    def restore(self, model_dir, is_masker=False):
        """Restore policy's networks from a saved model."""
        policy_actor_state_dict = torch.load(str(self.model_dir) + '/actor.pt')
        policy_critic_state_dict = torch.load(str(self.model_dir) + '/critic.pt')
        if is_masker:
            self.policy_masker.actor.load_state_dict(policy_actor_state_dict)
            self.policy_masker.critic.load_state_dict(policy_critic_state_dict)
        else:
            self.policy.actor.load_state_dict(policy_actor_state_dict)
            self.policy.critic.load_state_dict(policy_critic_state_dict)

    def log_train(self, train_infos, total_num_steps):
        """
        Log training info.
        :param train_infos: (dict) information about training update.
        :param total_num_steps: (int) total number of training env steps.
        """
        for k, v in train_infos.items():
            if self.use_wandb:
                wandb.log({k: v}, step=total_num_steps)
            else:
                self.writter.add_scalars(k, {k: v}, total_num_steps)

    def log_env(self, env_infos, total_num_steps):
        """
        Log env info.
        :param env_infos: (dict) information about env state.
        :param total_num_steps: (int) total number of training env steps.
        """
        for k, v in env_infos.items():
            if len(v) > 0:
                if self.use_wandb:
                    wandb.log({k: np.mean(v)}, step=total_num_steps)
                else:
                    self.writter.add_scalars(k, {k: np.mean(v)}, total_num_steps)
