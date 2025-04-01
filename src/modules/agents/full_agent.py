import torch as th
import torch.nn as nn
import torch.nn.functional as F


class FullAgent(nn.Module):
    def __init__(self, obs_dim, state_dim, args):
        super(FullAgent, self).__init__()
        self.args = args

        self.fc1_obs = nn.Linear(obs_dim, args.hidden_dim)
        self.fc1_state = nn.Linear(state_dim, args.hidden_dim)
        self.fc2 = nn.Linear(args.hidden_dim * 2, args.hidden_dim)
        self.fc3 = nn.Linear(args.hidden_dim, args.n_actions)

    def forward(self, inputs):
        """
        inputs: [obs, state]
        """
        obs, state = inputs
        x_obs = F.relu(self.fc1_obs(obs))
        x_state = F.relu(self.fc1_state(state))
        x = th.cat([x_obs, x_state], dim=-1)
        x = F.relu(self.fc2(x))
        q = self.fc3(x)
        return q