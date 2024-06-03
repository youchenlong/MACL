import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, input_shape, rnn_hidden_dim):
        super(Encoder, self).__init__()
        self.rnn_hidden_dim = rnn_hidden_dim
        self.fc = nn.Linear(input_shape, rnn_hidden_dim)
        self.rnn = nn.GRUCell(rnn_hidden_dim, rnn_hidden_dim)
    
    def forward(self, inputs, hidden_states):
        """
        inputs: [bs * n_agents, input_shape]
        hidden_states: [bs, n_agents, rnn_hidden_dim]
        """
        x = F.relu(self.fc(inputs))
        h_in = hidden_states.reshape(-1, self.rnn_hidden_dim)
        h = self.rnn(x, h_in)
        return h

    def init_hidden(self):
        return self.fc.weight.new(1, self.rnn_hidden_dim).zero_()

class MACLAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(MACLAgent, self).__init__()
        self.args = args
        self.input_shape = input_shape
        # temporal encoder
        self.encoder = Encoder(input_shape, args.rnn_hidden_dim)
        # policy network
        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.n_actions)

    def init_hidden(self):
        # make hidden states on same device as model
        return self.encoder.init_hidden()

    def forward(self, inputs, hidden_states):
        """
        inputs: [bs * n_agents, input_shape]
        hidden_states: [bs, n_agents, rnn_hidden_dim]
        """
        h = self.encoder(inputs, hidden_states)
        q = self.fc2(h)
        return q, h