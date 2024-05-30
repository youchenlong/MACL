import copy
from components.episode_buffer import EpisodeBatch
from modules.mixers.vdn import VDNMixer
from modules.mixers.qmix import QMixer
import torch as th
import torch.nn as nn
from torch.optim import RMSprop
from utils.attention import MultiHeadAttention
from utils.contrastive import infoNCE_loss


class MACLLearner:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.mac = mac
        self.logger = logger

        self.batch_size = args.batch_size
        self.n_agents = args.n_agents
        self.input_shape = self.mac._get_input_shape(scheme)

        # agent-level encoder
        self.attn = MultiHeadAttention(args.num_heads, args.attn_dim, args.softTemperature, self.input_shape, self.input_shape, self.input_shape, verbose=True, isSoftmax=args.isSoftmax)

        # projector
        self.projector = nn.Linear(args.rnn_hidden_dim + args.attn_dim, args.consensus_dim)

        # self.params = list(mac.parameters())
        self.params = list(mac.parameters()) + list(self.attn.parameters()) + list(self.projector.parameters())

        self.last_target_update_episode = 0

        self.mixer = None
        if args.mixer is not None:
            if args.mixer == "vdn":
                self.mixer = VDNMixer()
            elif args.mixer == "qmix":
                self.mixer = QMixer(args)
            else:
                raise ValueError("Mixer {} not recognised.".format(args.mixer))
            self.params += list(self.mixer.parameters())
            self.target_mixer = copy.deepcopy(self.mixer)

        self.optimiser = RMSprop(params=self.params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)

        # a little wasteful to deepcopy (e.g. duplicates action selector), but should work for any MAC
        self.target_mac = copy.deepcopy(mac)

        self.log_stats_t = -self.args.learner_log_interval - 1

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        # Get the relevant quantities
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]

        # Calculate estimated Q-Values
        mac_out = []
        temporal_features = []
        agent_level_features = []
        self.mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            agent_outs, hidden_states, agent_inputs = self.mac.forward(batch, t=t)
            temporal_feature = self.get_temporal_feature(hidden_states)
            temporal_features.append(temporal_feature)
            agent_level_feature = self.get_agent_level_feature(agent_inputs)
            agent_level_features.append(agent_level_feature)
            mac_out.append(agent_outs)
        mac_out = th.stack(mac_out, dim=1)  # Concat over time
        temporal_features = th.stack(temporal_features, dim=1) # [bs, ts, n_agents, rnn_hidden_dim]
        agent_level_features = th.stack(agent_level_features, dim=1) # [bs, ts, n_agents, attn_dim]
        
        # fusion and projection
        fusion_features = self.fusion(temporal_features, agent_level_features)
        consensus_features = self.projector(fusion_features)

        # Pick the Q-Values for the actions taken by each agent
        chosen_action_qvals = th.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3)  # Remove the last dim

        # Calculate the Q-Values necessary for the target
        target_mac_out = []
        self.target_mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            target_agent_outs, _, _ = self.target_mac.forward(batch, t=t)
            target_mac_out.append(target_agent_outs)

        # We don't need the first timesteps Q-Value estimate for calculating targets
        target_mac_out = th.stack(target_mac_out[1:], dim=1)  # Concat across time

        # Mask out unavailable actions
        target_mac_out[avail_actions[:, 1:] == 0] = -9999999

        # Max over target Q-Values
        if self.args.double_q:
            # Get actions that maximise live Q (for double q-learning)
            mac_out_detach = mac_out.clone().detach()
            mac_out_detach[avail_actions == 0] = -9999999
            cur_max_actions = mac_out_detach[:, 1:].max(dim=3, keepdim=True)[1]
            target_max_qvals = th.gather(target_mac_out, 3, cur_max_actions).squeeze(3)
        else:
            target_max_qvals = target_mac_out.max(dim=3)[0]

        # Mix
        if self.mixer is not None:
            chosen_action_qvals = self.mixer(chosen_action_qvals, batch["state"][:, :-1])
            target_max_qvals = self.target_mixer(target_max_qvals, batch["state"][:, 1:])

        # Calculate 1-step Q-Learning targets
        targets = rewards + self.args.gamma * (1 - terminated) * target_max_qvals

        # Td-error
        td_error = (chosen_action_qvals - targets.detach())

        mask = mask.expand_as(td_error)

        # 0-out the targets that came from padded data
        masked_td_error = td_error * mask

        # Normal L2 td_loss, take mean over actual data
        td_loss = (masked_td_error ** 2).sum() / mask.sum()

        # calculate the contrastive loss  
        contrastive_loss = self.calculate_contrastive_loss(consensus_features)

        loss = td_loss + self.args.contrastive_loss_weight * contrastive_loss

        # Optimise
        self.optimiser.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
        self.optimiser.step()

        if (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_episode = episode_num

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("loss", loss.item(), t_env)
            self.logger.log_stat("td_loss", td_loss.item(), t_env)
            self.logger.log_stat("contrastive_loss", contrastive_loss.item(), t_env)
            self.logger.log_stat("grad_norm", grad_norm, t_env)
            mask_elems = mask.sum().item()
            self.logger.log_stat("td_error_abs", (masked_td_error.abs().sum().item()/mask_elems), t_env)
            self.logger.log_stat("q_taken_mean", (chosen_action_qvals * mask).sum().item()/(mask_elems * self.args.n_agents), t_env)
            self.logger.log_stat("target_mean", (targets * mask).sum().item()/(mask_elems * self.args.n_agents), t_env)
            self.log_stats_t = t_env
            # for name, param in self.mac.agent.named_parameters():
            #     print("parameters in mac:", name, param)
            # for name, param in self.attn.named_parameters():
            #     print("parameters in attn:", name, param)
            # for name, param in self.projector.named_parameters():
            #     print("parameters in projector", name, param)

    def get_temporal_feature(self, hidden_states):
        """
        hidden_states: [bs, n_agents, rnn_hidden_dim]
        """
        return hidden_states.view(self.batch_size, self.n_agents, -1) # [bs, n_agents, rnn_hidden_dim]

    def get_agent_level_feature(self, agent_inputs):
        """
        agent_inputs: [bs, n_agents, input_shape]
        """
        q = agent_inputs.reshape(self.batch_size * self.n_agents, 1, self.input_shape) # [bs * n_agents, 1, input_shape]
        k = agent_inputs.unsqueeze(1).expand(-1, self.n_agents, -1, -1).reshape(self.batch_size * self.n_agents, self.n_agents, self.input_shape) # [bs * n_agents, n_agents, input_shape]
        v = agent_inputs.unsqueeze(1).expand(-1, self.n_agents, -1, -1).reshape(self.batch_size * self.n_agents, self.n_agents, self.input_shape) # [bs * n_agents, n_agents, input_shape]
        agent_level_feature = self.attn(q, k, v)
        return agent_level_feature.view(self.batch_size, self.n_agents, -1) # [bs, n_agents, attn_dim]

    def fusion(self, temporal_features, agent_level_features):
        """
        temporal_features: [bs, ts, n_agents, rnn_hidden_dim]
        agent_level_features: [bs, ts, n_agents, attn_dim]
        """
        fusion_features = th.cat([temporal_features, agent_level_features], dim=-1) # [bs, ts, n_agents, rnn_hidden_dim + attn_dim]
        return fusion_features

    def calculate_contrastive_loss(self, consensus_features):
        """
        consensus_features: [bs, ts, n_agents, consensus_dim]
        """
        bs, ts, n_agents, consensus_dim = consensus_features.shape
        consensus_features = consensus_features.view(-1, consensus_dim) # [bs * ts * n_agents, consensus_dim]
        # mask
        labels = th.arange(bs * ts, device=consensus_features.device).repeat_interleave(n_agents)
        positive_mask = labels.unsqueeze(0) == labels.unsqueeze(1) # [bs * ts * n_agents, bs * ts * n_agents]
        # contrastive loss
        contrastive_loss = infoNCE_loss(consensus_features, positive_mask, self.args.contrasTemperature)
        return contrastive_loss

    def _update_targets(self):
        self.target_mac.load_state(self.mac)
        if self.mixer is not None:
            self.target_mixer.load_state_dict(self.mixer.state_dict())
        self.logger.console_logger.info("Updated target network")

    def cuda(self):
        self.mac.cuda()
        self.target_mac.cuda()
        self.attn.cuda()
        self.projector.cuda()    
        if self.mixer is not None:
            self.mixer.cuda()
            self.target_mixer.cuda()

    def save_models(self, path):
        self.mac.save_models(path)
        th.save(self.attn.state_dict(), "{}/attn.th".format(path))
        th.save(self.projector.state_dict(), "{}/projector.th".format(path))
        if self.mixer is not None:
            th.save(self.mixer.state_dict(), "{}/mixer.th".format(path))
        th.save(self.optimiser.state_dict(), "{}/opt.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        # Not quite right but I don't want to save target networks
        self.target_mac.load_models(path)
        self.attn.load_state_dict(th.load("{}/attn.th".format(path), map_location=lambda storage, loc: storage))
        self.projector.load_state_dict(th.load("{}/projector.th".format(path), map_location=lambda storage, loc: storage))
        if self.mixer is not None:
            self.mixer.load_state_dict(th.load("{}/mixer.th".format(path), map_location=lambda storage, loc: storage))
        self.optimiser.load_state_dict(th.load("{}/opt.th".format(path), map_location=lambda storage, loc: storage))
