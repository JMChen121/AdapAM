import torch.nn as nn
import torch.nn.functional as F
import torch as th
import numpy as np
import torch.nn.init as init
from utils.th_utils import orthogonal_init_
from torch.nn import LayerNorm

class NRNNAgentSAIA(nn.Module):
    def __init__(self, input_shapes, args):
        super(NRNNAgentSAIA, self).__init__()
        self.args = args

        self.state_fc = nn.Linear(input_shapes[0], args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.ob_fc = nn.Linear(input_shapes[0], args.rnn_hidden_dim)
        self.action1_fc = nn.Linear(args.rnn_hidden_dim, args.n_actions[0])
        self.action2_fc = nn.Linear(args.rnn_hidden_dim, args.n_actions[1])

        if getattr(args, "use_layer_norm", False):
            raise NotImplementedError

        if getattr(args, "use_orthogonal", False):
            raise NotImplementedError

    def init_hidden(self):
        # make hidden states on same device as model
        return self.state_fc.weight.new(1, self.args.rnn_hidden_dim).zero_(), self.ob_fc.weight.new(1, self.args.rnn_hidden_dim).zero_()

    def forward(self, inputs, hidden_state):
        b, a, e = inputs.size()

        inputs = inputs.view(-1, e)
        x = F.relu(self.fc1(inputs), inplace=True)
        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        hh = self.rnn(x, h_in)

        if getattr(self.args, "use_layer_norm", False):
            q1 = self.fc2(self.layer_norm(hh))
            q2 = self.fc3(self.layer_norm(hh))
        else:
            q1 = self.fc2(hh)
            q2 = self.fc3(hh)

        return q1.view(b, a, -1), q2.view(b, a, -1), hh.view(b, a, -1)