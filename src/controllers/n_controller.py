from modules.agents import REGISTRY as agent_REGISTRY
from components.action_selectors import REGISTRY as action_REGISTRY
from .basic_controller import BasicMAC
import torch as th
from utils.rl_utils import RunningMeanStd
import numpy as np


# This multi-agent controller shares parameters between agents
class NMAC(BasicMAC):
    def __init__(self, scheme, groups, args):
        super(NMAC, self).__init__(scheme, groups, args)
        self.last_hidden_states = None

    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False, is_attack=False):
        # Only select actions for the selected batch elements in bs
        avail_actions = ep_batch["avail_actions"][:, t_ep]
        qvals = self.forward(ep_batch, t_ep, test_mode=test_mode, is_attack=is_attack)
        chosen_actions = self.action_selector.select_action(qvals[bs], avail_actions[bs], t_env, test_mode=test_mode)

        self.crt_agent_outs = qvals
        self.crt_chosen_actions = chosen_actions
        return chosen_actions

    def forward(self, ep_batch, t, test_mode=False, is_attack=False):
        if test_mode:
            self.agent.eval()

        agent_inputs = self._build_inputs(ep_batch, t)

        if is_attack:
            agent_outs, _ = self.agent(agent_inputs, self.last_hidden_states)
        else:
            self.last_hidden_states = self.hidden_states
            agent_outs, self.hidden_states = self.agent(agent_inputs, self.hidden_states)

        return agent_outs
