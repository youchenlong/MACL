import torch as th
import torch.nn as nn
import torch.nn.functional as F

"""
1.简单拼接
2.双分支融合
3.门控机制
"""

# class FullAgent(nn.Module):
#     def __init__(self, obs_dim, state_dim, args):
#         super(FullAgent, self).__init__()
#         self.args = args
#         self.fc1 = nn.Linear(obs_dim + state_dim, args.rnn_hidden_dim)
#         self.rnn_hidden_dim = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
#         self.fc2 = nn.Linear(args.rnn_hidden_dim, args.n_actions)

#     def init_hidden(self):
#         return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()

#     def forward(self, inputs, hidden_state):
#         """
#         inputs: [obs, state]
#         """
#         obs, state = inputs
#         x = F.relu(self.fc1(th.cat([obs, state], dim=-1)))
#         h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
#         h = self.rnn(x, h_in)
#         q = self.fc2(h)
#         return q, h
    

# class FullAgent(nn.Module):
#     def __init__(self, obs_dim, state_dim, args):
#         super(FullAgent, self).__init__()
#         self.args = args
#         self.fc1_obs = nn.Linear(obs_dim, args.rnn_hidden_dim)
#         self.rnn_obs = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
#         self.fc2_obs = nn.Linear(args.rnn_hidden_dim, args.hidden_dim)
#         self.fc1_state = nn.Linear(state_dim, args.hidden_dim)
#         self.fc2_state = nn.Linear(args.hidden_dim, args.hidden_dim)
#         self.fc_fuse = nn.Linear(args.hidden_dim * 2, args.n_actions)
    
#     def init_hidden(self):
#         return self.fc1_obs.weight.new(1, self.args.rnn_hidden_dim).zero_()
    
#     def forward(self, inputs, hidden_state):
#         """
#         inputs: [obs, state]
#         """
#         obs, state = inputs

#         h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
#         h = self.rnn_obs(F.relu(self.fc1_obs(obs)), h_in)
#         x_obs = self.fc2_obs(h)

#         x_state = F.relu(self.fc1_state(state))
#         x_state = self.fc2_state(x_state)

#         q = self.fc_fuse(th.cat([x_obs, x_state], dim=-1))
#         return q, h


class FullAgent(nn.Module):
    def __init__(self, obs_dim, state_dim, args):
        super(FullAgent, self).__init__()
        self.args = args

        self.fc1_obs = nn.Linear(obs_dim, args.rnn_hidden_dim)
        self.rnn_obs = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc2_obs = nn.Linear(args.rnn_hidden_dim, args.hidden_dim)
        self.fc1_state = nn.Linear(state_dim, args.hidden_dim)
        self.fc2_state = nn.Linear(args.hidden_dim, args.hidden_dim)
        self.fc_gate = nn.Linear(args.hidden_dim * 2, args.hidden_dim)
        self.fc_final = nn.Linear(args.hidden_dim, args.n_actions)

    def init_hidden(self):
        return self.fc1_obs.weight.new(1, self.args.rnn_hidden_dim).zero_()

    def forward(self, inputs, hidden_state):
        """
        inputs: [obs, state]
        """
        obs, state = inputs

        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        h = self.rnn_obs(F.relu(self.fc1_obs(obs)), h_in)
        x_obs = self.fc2_obs(h)
        
        x_state = F.relu(self.fc1_state(state))
        x_state = self.fc2_state(x_state)

        gate = th.sigmoid(self.fc_gate(th.cat([x_obs, x_state], dim=-1)))
        x = gate * h + (1 - gate) * x_state
        q = self.fc_final(x)
        return q, h