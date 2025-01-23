import torch as th
from torch.optim import Adam
import torch.nn.functional as F

from components.episode_buffer import EpisodeBatch
from controllers.n_controller_saia import QNetwork


def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


class SACLearnerSAIA:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.mac = mac
        self.logger = logger

        self.device = th.device('cuda' if args.use_cuda  else 'cpu')
        self.params = list(mac.parameters())

        self.log_stats_t = -self.args.learner_log_interval - 1

        '''for SAC'''
        self.gamma = args.gamma
        self.alpha = args.alpha

        self.policy_type = args.policy
        self.target_update_interval = args.target_update_interval_sac
        self.tau = args.tau if self.target_update_interval == 1 else 1
        self.automatic_entropy_tuning = args.automatic_entropy_tuning

        self.critic = QNetwork(scheme, args).to(device=self.device)
        self.critic_optim = Adam(self.critic.parameters(), lr=args.lr)
        self.params += list(self.critic.parameters())

        self.critic_target = QNetwork(scheme, args).to(self.device)
        hard_update(self.critic_target, self.critic)

        if self.policy_type == "Gaussian":
            if self.automatic_entropy_tuning is True:
                self.target_entropy = -th.prod(th.Tensor(scheme["actions2"]["vshape"]).to(self.device)).item()
                self.log_alpha = th.zeros(1, requires_grad=True, device=self.device)
                self.alpha_optim = Adam([self.log_alpha], lr=args.lr)

            self.policy = self.mac
            self.policy_optim = Adam(self.policy.parameters(), lr=args.lr)

        else:
            raise NotImplementedError
        
    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int, updates):
        # Sample a batch from memory
        state_batch = batch["state"][:, :-1]
        reward_batch = batch["reward"][:, :-1]
        action1_batch = batch["actions1"][:, :-1]
        action2_batch = batch["actions2"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask_batch = batch["filled"][:, :-1].float()
        mask_batch[:, 1:] = mask_batch[:, 1:] * (1 - terminated[:, :-1])

        '''Calculate estimated Q-Values for Updating critic network'''
        self.mac.train()
        self.mac.init_hidden(batch.batch_size)
        q_targets_list = []
        qf1_list = []
        qf2_list = []
        for next_t_ep in range(0, batch.max_seq_length):
            with th.no_grad():
                next_envs_not_terminated = [b_idx for b_idx, mask in enumerate(batch["filled"][:, next_t_ep]) if mask]
                vic_agent, vic_agent_q, next_action2, log_pi, _ = self.policy.select_actions(batch, next_t_ep, t_env, bs=next_envs_not_terminated)
                next_action1 = th.zeros(len(next_envs_not_terminated), self.args.n_actions[0], device=self.device).scatter_(-1, vic_agent, 1)  # set to 1 at dim=2
                if next_t_ep > 0:
                    # target critic, using next state of batch and next actions
                    qf1_next_target, qf2_next_target = self.critic_target(batch, next_action1, next_action2, next_t_ep, bs=next_envs_not_terminated)
                    min_qf_next_target = th.zeros(batch.batch_size, 1, device=self.device)
                    min_qf_next_target[next_envs_not_terminated] = th.min(qf1_next_target, qf2_next_target) - self.alpha * (vic_agent_q.gather(-1, vic_agent.unsqueeze(-1)).squeeze(-1) + log_pi)
                    q_targets = reward_batch[:, next_t_ep - 1] + mask_batch[:, next_t_ep - 1] * self.gamma * min_qf_next_target
                    q_targets_list.append(q_targets)
            # critic, using current state and actions in batch
            if next_t_ep > 0:
                envs_not_terminated = [b_idx for b_idx, mask in enumerate(batch["filled"][:, next_t_ep - 1]) if mask]
                action1 = th.zeros(len(envs_not_terminated), self.args.n_actions[0], device=self.device).scatter_(-1, action1_batch[envs_not_terminated, next_t_ep-1, 0], 1)
                qf1, qf2 = th.zeros(batch.batch_size, 1, device=self.device), th.zeros(batch.batch_size, 1, device=self.device)
                qf1[envs_not_terminated], qf2[envs_not_terminated] = self.critic(batch, action1, action2_batch[envs_not_terminated, next_t_ep-1], next_t_ep-1, bs=envs_not_terminated)  # Two Q-functions to mitigate positive bias in the policy improvement step
                qf1_list.append(qf1)
                qf2_list.append(qf2)
        all_q_targets = th.stack(q_targets_list, dim=0)
        all_qf1 = th.stack(qf1_list, dim=0)
        all_qf2 = th.stack(qf2_list, dim=0)
        qf1_loss = F.mse_loss(all_qf1, all_q_targets)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf2_loss = F.mse_loss(all_qf2, all_q_targets)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf_loss = qf1_loss + qf2_loss
        '''Update critic network'''
        self.critic_optim.zero_grad()
        qf_loss.backward()
        self.critic_optim.step()

        '''Update policy network'''
        self.mac.init_hidden(batch.batch_size)
        a1_q_list, a2_q_list = [], []
        policy_loss_list = []
        for t_ep in range(0, batch.max_seq_length-1):
            a1_q, a2_q = th.zeros(batch.batch_size, 1, device=self.device), th.zeros(batch.batch_size, 1, device=self.device)
            qf1_pi, qf2_pi = th.zeros(batch.batch_size, 1, device=self.device), th.zeros(batch.batch_size, 1, device=self.device)
            envs_not_terminated = [b_idx for b_idx, mask in enumerate(batch["filled"][:, t_ep]) if mask]

            vic_agent, vic_agent_q, action2, a2_q[envs_not_terminated], _ = self.policy.select_actions(batch, t_ep, t_env, bs=envs_not_terminated)
            action1 = th.zeros(len(envs_not_terminated), self.args.n_actions[0], device=self.device).scatter_(-1, vic_agent, 1)

            a1_q[envs_not_terminated] = vic_agent_q.gather(-1, vic_agent.unsqueeze(-1)).squeeze(-1)
            qf1_pi[envs_not_terminated], qf2_pi[envs_not_terminated] = self.critic(batch, action1, action2, t_ep, bs=envs_not_terminated)
            min_qf_pi = th.min(qf1_pi, qf2_pi)
            policy_loss_ = self.alpha * (a1_q + a2_q) - min_qf_pi  # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]
            policy_loss_list.append(policy_loss_)
            a1_q_list.append(a1_q)
            a2_q_list.append(a2_q)
        policy_loss = th.stack(policy_loss_list, dim=0).mean()
        self.policy_optim.zero_grad()
        policy_loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
        self.policy_optim.step()

        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

            self.alpha = self.log_alpha.exp()
            alpha_tlogs = self.alpha.clone()  # For TensorboardX logs
        else:
            alpha_loss = th.tensor(0.).to(self.device)
            alpha_tlogs = th.tensor(self.alpha)  # For TensorboardX logs

        if updates % self.target_update_interval == 0:
            soft_update(self.critic_target, self.critic, self.tau)
            # self.logger.console_logger.info("Updated target critic network")

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("loss_critic", qf_loss.item(), t_env)
            self.logger.log_stat("loss_policy", policy_loss.item(), t_env)
            self.logger.log_stat("grad_norm", grad_norm, t_env)
            mask_elems = mask_batch.sum().item()
            self.logger.log_stat("a1_q_taken_mean", th.stack(a1_q_list, dim=0).sum().item() / mask_elems, t_env)
            self.logger.log_stat("a2_q_taken_mean", th.stack(a2_q_list, dim=0).sum().item() / mask_elems, t_env)
            self.logger.log_stat("q_target_mean", all_q_targets.sum().item() / mask_elems, t_env)
            self.log_stats_t = t_env

        return qf1_loss.item(), qf2_loss.item(), policy_loss.item(), alpha_loss.item(), alpha_tlogs.item()


    def cuda(self):
        self.mac.cuda()
        # self.target_mac.cuda()
        # if self.mixer is not None:
        #     self.mixer.cuda()
        #     self.target_mixer.cuda()
            
    def save_models(self, path):
        self.mac.save_models(path)
        # if self.mixer is not None:
        #     th.save(self.mixer.state_dict(), "{}/mixer.th".format(path))
        # th.save(self.optimiser.state_dict(), "{}/opt.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        # Not quite right but I don't want to save target networks
        # self.target_mac.load_models(path)
        # if self.mixer is not None:
        #     self.mixer.load_state_dict(th.load("{}/mixer.th".format(path), map_location=lambda storage, loc: storage))
        # self.optimiser.load_state_dict(th.load("{}/opt.th".format(path), map_location=lambda storage, loc: storage))
