from .basic_controller_aia import BasicMACAIA


# This multi-agent controller shares parameters between agents
def select_most_important(agent_inputs, avail_actions, t_env, test_mode=False):
    pass


class NMACAIA(BasicMACAIA):
    def __init__(self, scheme, groups, args):
        super(NMACAIA, self).__init__(scheme, groups, args)
        self.last_hidden_states = None

    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False, is_attack=False):
        # Only select actions for the selected batch elements in bs
        avail_actions1 = ep_batch["avail_actions1"][:, t_ep]
        avail_actions2 = ep_batch["avail_actions2"][:, t_ep]
        qvals1, qvals2 = self.forward(ep_batch, t_ep, test_mode=test_mode, is_attack=is_attack)

        # chosen_actions1 = self.action_selector.select_action(qvals1[bs], avail_actions1[bs], t_env, test_mode=test_mode)
        chosen_actions1, agents_importance = self.important_selector.select_action(qvals1[bs], avail_actions1[bs], t_env, test_mode=test_mode)
        chosen_actions2 = self.action_selector.select_action(qvals2[bs], avail_actions2[bs], t_env, test_mode=test_mode)

        self.crt_agent_outs1 = qvals1
        self.crt_agent_outs2 = qvals2
        self.crt_chosen_actions1 = chosen_actions1
        self.crt_chosen_actions2 = chosen_actions2
        return chosen_actions1, chosen_actions2, agents_importance

    def forward(self, ep_batch, t, test_mode=False, is_attack=False):
        if test_mode:
            self.agent.eval()

        agent_inputs = self._build_inputs(ep_batch, t)

        if is_attack:
            agent_outs1, agent_outs2, _ = self.agent(agent_inputs, self.last_hidden_states)
        else:
            self.last_hidden_states = self.hidden_states
            agent_outs1, agent_outs2, self.hidden_states = self.agent(agent_inputs, self.hidden_states)

        return agent_outs1, agent_outs2
