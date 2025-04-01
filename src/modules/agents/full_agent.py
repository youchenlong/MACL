import torch.nn as nn
import torch.nn.functional as F


class FullAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(FullAgent, self).__init__()
        self.args = args

        self.fc1 = nn.Linear(input_shape, args.hidden_dim_1)
        self.fc2 = nn.Linear(args.hidden_dim_1, args.hidden_dim_2)
        self.fc3 = nn.Linear(args.hidden_dim_2, args.n_actions)

    def forward(self, inputs):
        x = F.relu(self.fc1(inputs))
        x = F.relu(self.fc2(x))
        q = self.fc3(x)
        return q