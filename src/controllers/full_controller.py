from modules.agents import REGISTRY as agent_REGISTRY
from components.action_selectors import REGISTRY as action_REGISTRY
import torch as th


class FullMAC:
    def __init__(self, scheme, groups, args):
        self.n_agents = args.n_agents
        self.args = args
        obs_dim = self._get_obs_dim(scheme)
        state_dim = self._get_state_dim(scheme)
        self._build_agents(obs_dim, state_dim)
        self.agent_output_type = args.agent_output_type

        self.action_selector = action_REGISTRY[args.action_selector](args)

    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):
        # Only select actions for the selected batch elements in bs
        avail_actions = ep_batch["avail_actions"][:, t_ep]
        agent_outputs = self.forward(ep_batch, t_ep, test_mode=test_mode)
        chosen_actions = self.action_selector.select_action(agent_outputs[bs], avail_actions[bs], t_env, test_mode=test_mode)
        return chosen_actions

    def forward(self, ep_batch, t, test_mode=False):
        agent_inputs = self._build_inputs(ep_batch, t)
        agent_actions = ep_batch["avail_actions"][:, t]
        agent_outs = self.agent(agent_inputs)
        if self.agent_output_type == "pi_logits":
            if getattr(self.args, "mask_before_softmax", True):
                reshaped_avail_actions = agent_actions.reshape(ep_batch.batch_size * self.n_agents, -1)
                agent_outs[reshaped_avail_actions == 0] = -1e10
            agent_outs = th.nn.functional.softmax(agent_outs, dim=-1)
            if not test_mode:
                epsilon_action_num = agent_outs.size(-1)
                if getattr(self.args, "mask_before_softmax", True):
                    epsilon_action_num = reshaped_avail_actions.sum(dim=1, keepdim=True).float()
                agent_outs = ((1 - self.action_selector.epsilon) * agent_outs
                              + th.ones_like(agent_outs) * self.action_selector.epsilon / epsilon_action_num)
                if getattr(self.args, "mask_before_softmax", True):
                    agent_outs[reshaped_avail_actions == 0] = 0.0
        return agent_outs.view(ep_batch.batch_size, self.n_agents, -1)

    def init_hidden(self, batch_size):
        pass

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

    def _build_agents(self, obs_dim, state_dim):
        self.agent = agent_REGISTRY[self.args.agent](obs_dim, state_dim, self.args)

    def _build_inputs(self, batch, t):
        bs = batch.batch_size
        obs, state = [], []
        obs.append(batch["obs"][:, t])
        state.append(batch["state"][:, t].unsqueeze(1).expand(-1, self.n_agents, -1))
        if self.args.obs_last_action:
            if t == 0:
                obs.append(th.zeros_like(batch["actions_onehot"][:, t]))
                state.append(th.zeros_like(batch["actions_onehot"][:, t]))
            else:
                obs.append(batch["actions_onehot"][:, t-1])
                state.append(batch["actions_onehot"][:, t-1])
        if self.args.obs_agent_id:
            obs.append(th.eye(self.n_agents, device=batch.device).unsqueeze(0).expand(bs, -1, -1))
            state.append(th.eye(self.n_agents, device=batch.device).unsqueeze(0).expand(bs, -1, -1))
        obs = th.cat([x.reshape(bs*self.n_agents, -1) for x in obs], dim=1)
        state = th.cat([x.reshape(bs*self.n_agents, -1) for x in state], dim=1)
        inputs = [obs, state]
        return inputs

    def _get_obs_dim(self, scheme):
        obs_dim = scheme["obs"]["vshape"]
        if self.args.obs_last_action:
            obs_dim += scheme["actions_onehot"]["vshape"][0]
        if self.args.obs_agent_id:
            obs_dim += self.n_agents
        return obs_dim

    def _get_state_dim(self, scheme):
        state_dim = scheme["state"]["vshape"]
        if self.args.obs_last_action:
            state_dim += scheme["actions_onehot"]["vshape"][0]
        if self.args.obs_agent_id:
            state_dim += self.n_agents
        return state_dim