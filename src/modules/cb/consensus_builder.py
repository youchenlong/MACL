import copy
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from components.epsilon_schedules import LinearSchedule

class ConsensusBuilder(nn.Module):
    def __init__(self, encoder, args, input_shape):
        super(ConsensusBuilder, self).__init__()
        self.args = args
        self.input_shape = input_shape

        # the same encoder used in policy learning
        self.online_encoder = encoder
        # transition model: extract temperal predictive feature
        self.hidden_state_decoder = nn.Sequential(
            nn.Linear(self.args.rnn_hidden_dim + self.args.n_actions + self.args.state_shape, self.args.rnn_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.args.rnn_hidden_dim, self.args.rnn_hidden_dim)
        )
        self.reward_decoder = nn.Sequential(
            nn.Linear(self.args.rnn_hidden_dim + self.args.n_actions + self.args.state_shape, self.args.rnn_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.args.rnn_hidden_dim, 1)
        )
        # nonlinear projector
        self.online_projector = nn.Sequential(
            nn.Linear(self.args.rnn_hidden_dim, self.args.consensus_dim * 2),
            nn.ReLU(),
            nn.Linear(self.args.consensus_dim * 2, self.args.consensus_dim)
        )

        # similar to online encoder and online projector, but stop backpropagation of gradients through the target network
        self.target_encoder = copy.deepcopy(encoder)
        self.target_projector = nn.Sequential(
            nn.Linear(self.args.rnn_hidden_dim, self.args.consensus_dim * 2),
            nn.ReLU(),
            nn.Linear(self.args.consensus_dim * 2, self.args.consensus_dim)
        )

        self.schedule = LinearSchedule(args.tau_start, args.tau_finish, args.tau_anneal_time)
        self.tau = self.schedule.eval(0)
    
    def calc_student(self, inputs, hidden_states):
        """
        inputs: [bs, ts, n_agents, input_shape]
        hidden_states: [bs, ts, n_agents, rnn_hidden_dim]
        """     

        representation = self.online_encoder(inputs.reshape(-1, self.input_shape), hidden_states.reshape(-1, self.args.rnn_hidden_dim)) # [bs * ts * n_agents, rnn_hidden_dim]
        projection = self.online_projector(representation) # [bs * ts * n_agents, consensus_dim]
        return projection
    
    def calc_teacher(self, inputs, hidden_states):
        """
        inputs: [bs, ts, n_agents, input_shape]
        hidden_states: [bs, ts, n_agents, rnn_hidden_dim]
        """

        representation = self.target_encoder(inputs.reshape(-1, self.input_shape), hidden_states.reshape(-1, self.args.rnn_hidden_dim)) # [bs * ts * n_agents, rnn_hidden_dim]
        projection = self.target_projector(representation) # [bs * ts * n_agents, consensus_dim]
        return projection

    def parameters(self):
        return list(self.online_encoder.parameters()) + list(self.hidden_state_decoder.parameters()) + list(self.reward_decoder.parameters()) + list(self.online_projector.parameters())

    def update_targets(self, t_env):
        self.tau = self.schedule.eval(t_env)

        for param_o, param_t in zip(self.online_encoder.parameters(), self.target_encoder.parameters()):
            param_t.data = param_t.data * self.tau + param_o.data * (1. - self.tau)

        for param_o, param_t in zip(self.online_projector.parameters(), self.target_projector.parameters()):   
            param_t.data = param_t.data * self.tau + param_o.data * (1. - self.tau)
    
    def save_models(self, path):
        th.save(self.online_encoder.state_dict(), "{}/online_encoder.th".format(path))
        th.save(self.hidden_state_decoder.state_dict(), "{}/hidden_state_decoder.th".format(path))
        th.save(self.reward_decoder.state_dict(), "{}/reward_decoder.th".format(path))
        th.save(self.online_projector.state_dict(), "{}/online_projector.th".format(path))

    def load_models(self, path):
        self.online_encoder.load_state_dict(th.load("{}/online_encoder.th".format(path), map_location=lambda storage, loc: storage))
        self.hidden_state_decoder.load_state_dict(th.load("{}/hidden_state_decoder.th".format(path), map_location=lambda storage, loc: storage))
        self.reward_decoder.load_state_dict(th.load("{}/reward_decoder.th".format(path), map_location=lambda storage, loc: storage))
        self.online_projector.load_state_dict(th.load("{}/online_projector.th".format(path), map_location=lambda storage, loc: storage))

