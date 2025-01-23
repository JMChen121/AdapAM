import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from components.action_selectors import REGISTRY as action_REGISTRY

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6


# Initialize Policy weights
def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)


class QNetwork(nn.Module):
    def __init__(self, scheme, args):
        super(QNetwork, self).__init__()

        self.input_shape = scheme["state"]["vshape"] + args.n_actions[0] + args.n_actions[1]
        self.hidden_dim1 = 128
        self.hidden_dim2 = 64

        # Q1 architecture
        self.linear1_1 = nn.Linear(self.input_shape, self.hidden_dim1)
        self.linear1_2 = nn.Linear(self.hidden_dim1, self.hidden_dim2)
        self.linear1_3 = nn.Linear(self.hidden_dim2, 1)

        # Q2 architecture
        self.linear2_1 = nn.Linear(self.input_shape, self.hidden_dim1)
        self.linear2_2 = nn.Linear(self.hidden_dim1, self.hidden_dim2)
        self.linear2_3 = nn.Linear(self.hidden_dim2, 1)

        self.apply(weights_init_)

    def forward(self, batch, actions1, actions2, t_ep, bs=slice(None)):
        state = batch["state"][bs, t_ep]
        x_i = torch.cat([state, actions1, actions2], -1)

        x1 = F.relu(self.linear1_1(x_i))
        x1 = F.relu(self.linear1_2(x1))
        x1 = self.linear1_3(x1)

        x2 = F.relu(self.linear2_1(x_i))
        x2 = F.relu(self.linear2_2(x2))
        x2 = self.linear2_3(x2)

        return x1, x2


class GaussianPolicy(nn.Module):
    # def __init__(self, num_inputs, num_actions, hidden_dim, action_space=None):
    def __init__(self, scheme, groups, args, action_space=None):
        super(GaussianPolicy, self).__init__()
        self.n_agents = args.n_agents
        self.args = args

        self.state_shape = scheme["state"]["vshape"]
        self.state_input_shape = self.state_shape + args.n_actions[0]
        self.obs_shape = scheme["obs"]["vshape"]
        self.ob_feature_dim = 16

        self.action_selector = action_REGISTRY[args.action_selector](args)

        self.hidden_states = None

        self.state_fc = nn.Linear(self.state_input_shape, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.ob_fc = nn.Linear(self.obs_shape, self.ob_feature_dim)

        # victim selection
        self.victim_fc = nn.Linear(args.rnn_hidden_dim, args.n_actions[0])
        # perturbation
        self.mean_linear = nn.Linear(args.rnn_hidden_dim + self.ob_feature_dim, args.n_actions[1])
        self.log_std_linear = nn.Linear(args.rnn_hidden_dim + self.ob_feature_dim, args.n_actions[1])

        self.apply(weights_init_)

        # action rescaling
        if action_space is None:
            self.action_scale = torch.tensor(1.)
            self.action_bias = torch.tensor(0.)
        else:
            self.action_scale = torch.FloatTensor(
                    (action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor(
                    (action_space.high + action_space.low) / 2.)

    def init_hidden(self, batch_size):
        self.hidden_states = self.state_fc.weight.new(1, self.args.rnn_hidden_dim).zero_()
        if self.hidden_states is not None:
            self.hidden_states = self.hidden_states.unsqueeze(0).expand(batch_size, 1, -1)  # bav

    def forward(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):
        ep_bs = ep_batch.batch_size
        '''build state inputs for selecting victim agent'''
        state_inputs = []
        state_inputs.append(ep_batch["state"][:, t_ep])  # b1av
        if t_ep == 0:
            state_inputs.append(torch.zeros_like(ep_batch["actions1_onehot"][:, t_ep]))
        else:
            state_inputs.append(ep_batch["actions1_onehot"][:, t_ep - 1])
        state_inputs = torch.cat([x.reshape(ep_bs, 1, -1) for x in state_inputs], dim=-1)
        '''select victim agent'''
        state_b, state_a, state_e = state_inputs.size()
        victim_inputs = state_inputs.view(-1, state_e)
        x = F.relu(self.state_fc(victim_inputs), inplace=True)
        h_in = self.hidden_states.reshape(-1, self.args.rnn_hidden_dim)
        hh = self.rnn(x, h_in)
        self.hidden_states = hh.view(state_b, state_a, -1)
        vic_agent_q = self.victim_fc(hh)
        vic_agent_q = vic_agent_q.view(state_b, state_a, -1)
        avail_agent = ep_batch["avail_actions1"][:, t_ep]
        chosen_agent = self.action_selector.select_action(vic_agent_q[bs], avail_agent[bs], t_env, test_mode=test_mode)
        '''build inputs for generating perturbation'''
        pert_inputs = []
        obs = []
        for i in range(chosen_agent.shape[0]):
            for j in range(chosen_agent.shape[1]):
                if chosen_agent[i][j] >= ep_batch["obs"][i, t_ep].shape[0]:
                    obs.append(torch.zeros_like(ep_batch["obs"][i, t_ep, 0]))
                else:
                    obs.append(ep_batch["obs"][i, t_ep, chosen_agent[i][j]])
        obs = torch.stack(obs, dim=0).unsqueeze(1)
        obs_features = F.relu(self.ob_fc(obs))
        hh_obs = torch.cat((hh[bs].view(len(bs), state_a, -1), obs_features), dim=-1)
        pert_inputs.append(hh_obs)
        pert_inputs = torch.cat([x.reshape(len(bs), 1, -1) for x in pert_inputs], dim=-1)
        '''generate perturbation'''
        pert_b, pert_a, pert_e = pert_inputs.size()
        x2 = pert_inputs.view(-1, pert_e)
        mean = self.mean_linear(x2)
        log_std = self.log_std_linear(x2)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return vic_agent_q, chosen_agent, mean, log_std

    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):
        if test_mode:
            self.eval()
        vic_agent_q, vic_agent, mean, log_std = self.forward(ep_batch, t_ep, t_env, bs=bs, test_mode=test_mode)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        perturbation = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + epsilon)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return vic_agent, vic_agent_q, perturbation, log_prob, mean

    def cuda(self, device=None):
        self.action_scale.cuda()
        self.action_bias.cuda()
        return super(GaussianPolicy, self).cuda()

    def save_models(self, path):
        torch.save(self.state_dict(), "{}/agent.th".format(path))

    def load_models(self, path):
        self.load_state_dict(torch.load("{}/agent.th".format(path), map_location=lambda storage, loc: storage))


class DeterministicPolicy(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim, action_space=None):
        super(DeterministicPolicy, self).__init__()
        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean = nn.Linear(hidden_dim, num_actions)
        self.noise = torch.Tensor(num_actions)

        self.apply(weights_init_)

        # action rescaling
        if action_space is None:
            self.action_scale = 1.
            self.action_bias = 0.
        else:
            self.action_scale = torch.FloatTensor(
                    (action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor(
                    (action_space.high + action_space.low) / 2.)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        mean = torch.tanh(self.mean(x)) * self.action_scale + self.action_bias
        return mean

    def sample(self, state):
        mean = self.forward(state)
        noise = self.noise.normal_(0., std=0.1)
        noise = noise.clamp(-0.25, 0.25)
        action = mean + noise
        return action, torch.tensor(0.), mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        self.noise = self.noise.to(device)
        return super(DeterministicPolicy, self).to(device)