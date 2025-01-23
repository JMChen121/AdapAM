from modules.agents import REGISTRY as agent_REGISTRY
from components.action_selectors import REGISTRY as action_REGISTRY
import torch as th


# This multi-agent controller shares parameters between agents
class BasicMACSAIA:
    def __init__(self, scheme, groups, args):
        self.n_agents = args.n_agents
        self.args = args
        input_shape1, input_shape2 = self._get_input_shape(scheme)
        self._build_agents([input_shape1, input_shape2])
        self.agent_output_type = args.agent_output_type

        self.important_selector = action_REGISTRY["multinomial"](args)
        self.action_selector = action_REGISTRY[args.action_selector](args)
        self.save_probs = getattr(self.args, 'save_probs', False)

        self.hidden_states = None

        """My Params"""
        self.crt_agent_outs1 = None
        self.crt_chosen_actions1 = None
        self.crt_agent_outs2 = None
        self.crt_chosen_actions2 = None

    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):
        # Only select actions for the selected batch elements in bs
        avail_actions1 = ep_batch["avail_actions1"][:, t_ep]
        avail_actions2 = ep_batch["avail_actions2"][:, t_ep]
        agent_outputs1, agent_outputs2 = self.forward(ep_batch, t_ep, test_mode=test_mode)
        chosen_actions1 = self.action_selector.select_action(agent_outputs1[bs], avail_actions1[bs], t_env, test_mode=test_mode)
        chosen_actions2 = self.action_selector.select_action(agent_outputs2[bs], avail_actions2[bs], t_env, test_mode=test_mode)
        return chosen_actions1, chosen_actions2

    def forward(self, ep_batch, t, test_mode=False):
        agent_inputs = self._build_inputs(ep_batch, t)
        avail_actions = ep_batch["avail_actions"][:, t]
        if test_mode:
            self.agent.eval()
        agent_outs1, agent_outs2, self.hidden_states = self.agent(agent_inputs, self.hidden_states)

        # Softmax the agent outputs if they're policy logits
        if self.agent_output_type == "pi_logits":

            if getattr(self.args, "mask_before_softmax", True):
                # Make the logits for unavailable actions very negative to minimise their affect on the softmax
                agent_outs1 = agent_outs1.reshape(ep_batch.batch_size * self.n_agents, -1)
                reshaped_avail_actions = avail_actions.reshape(ep_batch.batch_size * self.n_agents, -1)
                agent_outs1[reshaped_avail_actions == 0] = -1e5

            agent_outs1 = th.nn.functional.softmax(agent_outs1, dim=-1)
            
        return agent_outs1.view(ep_batch.batch_size, self.n_agents, -1), agent_outs2.view(ep_batch.batch_size, self.n_agents, -1)

    def init_hidden(self, batch_size):
        self.hidden_states = self.agent.init_hidden()
        if self.hidden_states is not None:
            self.hidden_states = self.hidden_states.unsqueeze(0).expand(batch_size, self.n_agents, -1)  # bav

    def parameters(self):
        return self.agent.parameters()

    def load_state(self, other_mac):
        self.agent.load_state_dict(other_mac.agent.state_dict())

    def cuda(self):
        self.agent.cuda()

    def save_models(self, path):
        th.save(self.agent.state_dict(), "{}/agent.th".format(path))

    def load_models(self, path):
        self.agent.load_state_dict(th.load("{}/agent.th".format(path), map_location=lambda storage, loc: storage))

    def _build_agents(self, input_shape):
        self.agent = agent_REGISTRY[self.args.agent](input_shape, self.args)

    def _build_inputs(self, batch, t):
        # Assumes homogenous agents with flat observations.
        # Other MACs might want to e.g. delegate building inputs to each agent
        bs = batch.batch_size
        inputs = []
        inputs.append(batch["obs"][:, t])  # b1av
        if self.args.obs_last_action:
            if t == 0:
                inputs.append(th.zeros_like(batch["actions1_onehot"][:, t]))
                inputs.append(th.zeros_like(batch["actions2_onehot"][:, t]))
            else:
                inputs.append(batch["actions1_onehot"][:, t-1])
                inputs.append(batch["actions2_onehot"][:, t-1])
        if self.args.obs_agent_id:
            inputs.append(th.eye(self.n_agents, device=batch.device).unsqueeze(0).expand(bs, -1, -1))

        inputs = th.cat([x.reshape(bs, self.n_agents, -1) for x in inputs], dim=-1)
        return inputs

    def _get_input_shape(self, scheme):
        input_shape1 = scheme["obs"]["vshape"]
        input_shape2 = scheme["state"]["vshape"] + scheme["obs"]["vshape"]

        return input_shape1, input_shape2
