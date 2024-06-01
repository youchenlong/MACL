import torch as th
import torch.nn as nn
from utils.attention import MultiHeadAttention

class ConsensusBuilder(nn.Module):
    def __init__(self, args):
        super(ConsensusBuilder, self).__init__()
        self.args = args
        self.online_encoder = MultiHeadAttention(self.args.num_heads, self.args.attn_dim, self.args.softTemperature, self.args.rnn_hidden_dim, self.args.rnn_hidden_dim, self.args.rnn_hidden_dim, verbose=True, isSoftmax=self.args.isSoftmax)
        self.online_projector = nn.Linear(self.args.rnn_hidden_dim + self.args.attn_dim, self.args.consensus_dim)

        self.target_encoder = MultiHeadAttention(self.args.num_heads, self.args.attn_dim, self.args.softTemperature, self.args.rnn_hidden_dim, self.args.rnn_hidden_dim, self.args.rnn_hidden_dim, verbose=True, isSoftmax=self.args.isSoftmax)
        self.target_projector = nn.Linear(self.args.rnn_hidden_dim + self.args.attn_dim, self.args.consensus_dim)

    def calc_student(self, inputs):
        """
        inputs: [bs * ts, n_agents, rnn_hidden_dim]
        """
        q = inputs.reshape(-1, 1, self.args.rnn_hidden_dim) # [bs * ts * n_agents, 1, rnn_hidden_dim]
        k = inputs.unsqueeze(1).expand(-1, self.args.n_agents, -1, -1).reshape(-1, self.args.n_agents, self.args.rnn_hidden_dim) # [bs * ts * n_agents, n_agents, rnn_hidden_dim]
        v = inputs.unsqueeze(1).expand(-1, self.args.n_agents, -1, -1).reshape(-1, self.args.n_agents, self.args.rnn_hidden_dim) # [bs * ts * n_agents, n_agents, rnn_hidden_dim]
        representation = self.online_encoder(q, k, v) # [bs * ts * n_agents, 1, attn_dim]
        projection = self.online_projector(th.cat([inputs.view(-1, self.args.rnn_hidden_dim), representation.view(-1, self.args.attn_dim)], dim=-1))
        return projection

    def calc_teacher(self, inputs):
        """
        inputs: [bs * ts, n_agents, rnn_hidden_dim]
        """
        q = inputs.reshape(-1, 1, self.args.rnn_hidden_dim) # [bs * ts * n_agents, 1, rnn_hidden_dim]
        k = inputs.unsqueeze(1).expand(-1, self.args.n_agents, -1, -1).reshape(-1, self.args.n_agents, self.args.rnn_hidden_dim) # [bs * ts * n_agents, n_agents, rnn_hidden_dim]
        v = inputs.unsqueeze(1).expand(-1, self.args.n_agents, -1, -1).reshape(-1, self.args.n_agents, self.args.rnn_hidden_dim) # [bs * ts * n_agents, n_agents, rnn_hidden_dim]
        representation = self.target_encoder(q, k, v) # [bs * ts * n_agents, attn_dim]
        projection = self.target_projector(th.cat([inputs.view(-1, self.args.rnn_hidden_dim), representation.view(-1, self.args.attn_dim)], dim=-1))
        return projection

    def parameters(self):
        return list(self.online_encoder.parameters()) + list(self.online_projector.parameters())

    def update_targets(self):
        for param_o, param_t in zip(self.online_encoder.parameters(), self.target_encoder.parameters()):
            param_t.data = param_t.data * self.args.tau + param_o.data * (1. - self.args.tau)

        for param_o, param_t in zip(self.online_projector.parameters(), self.target_projector.parameters()):   
            param_t.data = param_t.data * self.args.tau + param_o.data * (1. - self.args.tau)
    
    def save_models(self, path):
        th.save(self.online_encoder.state_dict(), "{}/online_encoder.th".format(path))
        th.save(self.online_projector.state_dict(), "{}/online_projector.th".format(path))

    def load_models(self, path):
        self.online_encoder.load_state_dict(th.load("{}/online_encoder.th".format(path), map_location=lambda storage, loc: storage))
        self.online_projector.load_state_dict(th.load("{}/online_projector.th".format(path), map_location=lambda storage, loc: storage))
