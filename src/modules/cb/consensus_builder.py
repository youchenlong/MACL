import torch as th
import torch.nn as nn
from modules.agents.macl_agent import Encoder

class ConsensusBuilder(nn.Module):
    def __init__(self, encoder, args, input_shape):
        super(ConsensusBuilder, self).__init__()
        self.args = args
        self.input_shape = input_shape
        self.online_encoder = encoder
        
        self.hidden_state_decoder = nn.Sequential(
            nn.Linear(self.args.rnn_hidden_dim + self.args.n_actions, self.args.rnn_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.args.rnn_hidden_dim, self.args.rnn_hidden_dim)
        )
        self.reward_decoder = nn.Sequential(
            nn.Linear(self.args.rnn_hidden_dim + self.args.n_actions, self.args.rnn_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.args.rnn_hidden_dim, 1)
        )

        self.online_projector = nn.Sequential(
            nn.Linear(self.args.rnn_hidden_dim, self.args.consensus_dim * 2),
            nn.ReLU(),
            nn.Linear(self.args.consensus_dim * 2, self.args.consensus_dim)
        )

        self.target_encoder = Encoder(input_shape, args.rnn_hidden_dim)
        self.target_projector = nn.Sequential(
            nn.Linear(self.args.rnn_hidden_dim, self.args.consensus_dim * 2),
            nn.ReLU(),
            nn.Linear(self.args.consensus_dim * 2, self.args.consensus_dim)
        )

    def calc_student(self, inputs, hidden_states, actions):
        """
        inputs: [bs * ts, n_agents, input_shape]
        hidden_states: [bs * ts, n_agents, rnn_hidden_dim]
        action: [bs * ts, n_agents, n_actions]
        """
        representation = self.online_encoder(inputs.view(-1, self.input_shape), hidden_states) # [bs * ts * n_agents, rnn_hidden_dim]
        predict_representation = self.hidden_state_decoder(th.cat([representation, actions.view(-1, self.args.n_actions)], dim=-1)) # [bs * ts * n_agents, rnn_hidden_dim]   
        predict_reward = self.reward_decoder(th.cat([representation, actions.view(-1, self.args.n_actions)], dim=-1)) # [bs * ts * n_agents, 1]
        projection = self.online_projector(predict_representation) # [bs * ts * n_agents, consensus_dim]
        return projection, predict_representation, predict_reward


    def calc_teacher(self, inputs, hidden_states):
        """
        inputs: [bs * ts, n_agents, input_shape]
        hidden_states: [bs * ts, n_agents, rnn_hidden_dim]
        """
        representation = self.target_encoder(inputs.view(-1, self.input_shape), hidden_states) # [bs * ts * n_agents, rnn_hidden_dim]
        projection = self.target_projector(representation)
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

