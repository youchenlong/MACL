import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ToMAgent(nn.Module):
    def __init__(self, obs_dim, args):
        super(ToMAgent, self).__init__()
        self.args = args
        self.obs_dim = obs_dim
        self.n_agents = args.n_agents
        self.n_goals = args.state_info_dict[self.args.env_args["map_name"]]["n_enemies"]
        
        # used for imagination 
        self.env_name = args.env
        self.n_allies = args.state_info_dict[self.args.env_args["map_name"]]["n_allies"]
        self.nf_move = args.state_info_dict[self.args.env_args["map_name"]]["nf_move"]
        self.nf_own = args.state_info_dict[self.args.env_args["map_name"]]["nf_own"]
        self.nf_al = args.state_info_dict[self.args.env_args["map_name"]]["nf_al"]
        self.nf_en = args.state_info_dict[self.args.env_args["map_name"]]["nf_en"]
        self.sight_range = args.state_info_dict[self.args.env_args["map_name"]]["sight_range"]
        self.is_protoss = args.state_info_dict[self.args.env_args["map_name"]]["is_protoss"]
        self.unit_type_bits = args.state_info_dict[self.args.env_args["map_name"]]["unit_type_bits"]

        # mental inference
        self.fc1_mental = nn.Linear(obs_dim + args.n_actions, args.rnn_hidden_dim)
        self.rnn_mental = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc2_mental = nn.Linear(args.rnn_hidden_dim, args.mental_dim)

        # goal inference
        self.fc_infer = nn.Sequential(nn.Linear(obs_dim + args.mental_dim, args.NN_HIDDEN_DIM),
                                      nn.BatchNorm1d(args.NN_HIDDEN_DIM),
                                      nn.LeakyReLU(),
                                      nn.Linear(args.NN_HIDDEN_DIM, self.n_goals))

        # goal generation
        self.fc_gen = nn.Linear(obs_dim + self.n_agents * self.n_goals, self.n_goals)

        # policy
        self.fc1 = nn.Linear(obs_dim + self.n_goals, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.n_actions)

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()

    def init_mental(self):
        self.o_mentals = None
        return self.fc1_mental.weight.new(1, self.args.rnn_hidden_dim).zero_()

    def forward(self, obs, actions, hidden_state, hidden_state_mental, mask):
        """
        obs: [bs, n_agents, obs_dim]
        actions: [bs, n_agents, n_actions]
        hidden_state: [bs, n_agents, rnn_hidden_dim]
        hidden_state_mental: [bs, n_agents * n_agents, rnn_hidden_dim]
        mask: [bs, n_agents, n_agents]
        """

        # observation imagination
        obs = obs.reshape(-1, self.obs_dim) # [bs*n_agents, obs_dim]
        imagined_obs = self.imagination(obs) # [bs*n_agents*n_agents, obs_dim]

        # goal inference
        _obs = obs.repeat(1, self.n_agents).reshape(-1, self.obs_dim) # [bs*n_agents*n_agents, obs_dim]
        if self.o_mentals is None:
            self.o_mentals = torch.zeros((_obs.size(0), self.args.mental_dim), device=obs.device) # [bs*n_agents*n_agents, mental_dim]
        o_goal = F.softmax(self.fc_infer(torch.cat([_obs, self.o_mentals], dim=-1)), dim=-1) # [bs*n_agents*n_agents, n_goals]
        _mask = mask.reshape(-1, 1) # [bs*n_agents*n_agents, 1]
        o_goal = o_goal * _mask # [bs*n_agents*n_agents, n_goals]

        # goal generation
        goal = F.softmax(self.fc_gen(torch.cat([obs, o_goal.reshape(-1, self.n_agents * self.n_goals)], dim=-1))) # [bs*n_agents, n_goals]

        # policy
        x = F.relu(self.fc1(torch.cat([obs, goal], dim=-1)))
        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        h = self.rnn(x, h_in)
        q = self.fc2(h)

        # mental inference
        _actions = actions.repeat(1, self.n_agents, 1).reshape(-1, self.args.n_actions) # [bs*n_agents*n_agents, n_actions]
        x_o_mentals = F.relu(self.fc1_mental(torch.cat([_obs, _actions], dim=-1))) # [bs*n_agents*n_agents, rnn_hidden_dim]
        h_in_o_mental = hidden_state_mental.reshape(-1, self.args.rnn_hidden_dim) # [bs*n_agents*n_agents, rnn_hidden_dim]
        h_o_mental = self.rnn_mental(x_o_mentals, h_in_o_mental) # [bs*n_agents*n_agents, rnn_hidden_dim]
        self.o_mentals = self.fc2_mental(h_o_mental) # [bs*n_agents*n_agents, mental_dim]
        self.o_mentals = self.o_mentals * _mask # [bs*n_agents*n_agents, mental_dim]

        return q, h, h_o_mental, goal, o_goal

    def imagination(self, obs):
        """
        obs: [bs * n_agents, obs_dim]
        """
        assert obs.size(-1) == self.nf_move + self.nf_own + self.nf_al * self.n_allies + self.nf_en * self.n_goals
        imagined_obs = []
        obs_clone = obs.clone().cpu().numpy() # [bs*n_agents, obs_dim]
        for idx, o in enumerate(obs_clone):
            agent_id = idx % self.n_agents
            if self.env_name == "sc2":
                _obs = self.imagined_obs_agent_sc2(agent_id, o, self.n_goals, self.n_allies, self.nf_move, self.nf_en, self.nf_al, self.nf_own, self.sight_range, self.is_protoss, self.unit_type_bits) # [n_agents, obs_dim]
            elif self.env_name == "sc2v2":
                _obs = self.imagined_obs_agent_sc2v2(agent_id, o, self.n_goals, self.n_allies, self.nf_move, self.nf_en, self.nf_al, self.nf_own, self.sight_range, self.is_protoss, self.unit_type_bits) # [n_agents, obs_dim]
            imagined_obs.append(torch.from_numpy(_obs).to(obs.device))
        return torch.stack(imagined_obs, dim=0) # [bs*n_agents*n_agents, obs_dim]
    

    def imagined_obs_agent_sc2(self, agent_id, obs, n_enemies, n_allies, move_feats_dim, enemy_feats_dim, ally_feats_dim, own_feats_dim, sight_range, is_protoss, unit_type_bits):
        move_feats = obs[:move_feats_dim]
        enemy_feats = obs[move_feats_dim:move_feats_dim+n_enemies*enemy_feats_dim].reshape(n_enemies, enemy_feats_dim)
        ally_feats = obs[move_feats_dim+n_enemies*enemy_feats_dim:move_feats_dim+n_enemies*enemy_feats_dim+n_allies*ally_feats_dim].reshape(n_allies, ally_feats_dim)
        own_feats = obs[move_feats_dim+n_enemies*enemy_feats_dim+n_allies*ally_feats_dim:move_feats_dim+n_enemies*enemy_feats_dim+n_allies*ally_feats_dim+own_feats_dim]
        
        imagined_move_feats = np.zeros((n_allies, move_feats_dim))
        imagined_enemy_feats = np.zeros((n_allies, n_enemies, enemy_feats_dim))
        imagined_ally_feats = np.zeros((n_allies, n_allies, ally_feats_dim))
        imagined_own_feats = np.zeros((n_allies, own_feats_dim))

        for a_idx in range(n_allies):
            a_visible = ally_feats[a_idx][0]
            if a_visible == 0:
                continue
            # imagined movement features
            imagined_move_feats[a_idx] = move_feats # meaningless
            # imagined enemy features
            a_agent_id = a_idx if a_idx < agent_id else a_idx + 1
            a_sight_range = sight_range
            a_position = ally_feats[a_idx][2:4]
            for e_idx in range(n_enemies):
                e_visible = enemy_feats[e_idx][0]
                if e_visible == 0:
                    continue
                e_position = enemy_feats[e_idx][2:4]
                if math.dist(e_position, a_position) * sight_range <= a_sight_range:
                    imagined_enemy_feats[a_idx][e_idx] = enemy_feats[e_idx]
                    imagined_enemy_feats[a_idx][e_idx][0] = 1 # visible
                    imagined_enemy_feats[a_idx][e_idx][1] = math.dist(e_position, a_position) / sight_range # distance
                    imagined_enemy_feats[a_idx][e_idx][2] = (e_position[0] - a_position[0]) / sight_range # relative_x
                    imagined_enemy_feats[a_idx][e_idx][3] = (e_position[1] - a_position[1]) / sight_range # relative_y
            # imagined ally features
            for b_idx in range(n_allies):
                b_visible = ally_feats[b_idx][0]
                if b_visible == 0:
                    continue
                b_position = ally_feats[b_idx][2:4]
                if a_idx == b_idx:
                    imagined_ally_feats[a_idx][b_idx][0] = 1 # visible
                    imagined_ally_feats[a_idx][b_idx][1] = ally_feats[a_idx][1] # distance
                    imagined_ally_feats[a_idx][b_idx][2] = -ally_feats[a_idx][2] # relative_x
                    imagined_ally_feats[a_idx][b_idx][3] = -ally_feats[a_idx][3] # relative_y
                    ind = 0
                    imagined_ally_feats[a_idx][b_idx][ind+4] = own_feats[ind] # health
                    ind += 1
                    if is_protoss:
                        imagined_ally_feats[a_idx][b_idx][ind+4] = own_feats[ind] # shield
                        ind += 1
                    if unit_type_bits > 0:
                        imagined_ally_feats[a_idx][b_idx][ind+4] = own_feats[ind] # unit_type
                    continue
                if math.dist(a_position, b_position) * sight_range <= a_sight_range:
                    imagined_ally_feats[a_idx][b_idx] = ally_feats[b_idx]
                    imagined_ally_feats[a_idx][b_idx][0] = 1 # visible
                    imagined_ally_feats[a_idx][b_idx][1] = math.dist(b_position, a_position) / sight_range # distance
                    imagined_ally_feats[a_idx][b_idx][2] = (b_position[0] - a_position[0]) / sight_range # relative_x
                    imagined_ally_feats[a_idx][b_idx][3] = (b_position[1] - a_position[1]) / sight_range # relative_y
            # imagined own features
            ind = 0
            imagined_own_feats[a_idx][ind] = ally_feats[a_idx][ind+4] # health
            ind += 1
            if is_protoss:
                imagined_own_feats[a_idx][ind] = ally_feats[a_idx][ind+4] # shield
                ind += 1
            if unit_type_bits > 0:
                imagined_own_feats[a_idx][ind] = ally_feats[a_idx][-1] # unit_type

        imagined_other_obs = np.concatenate([imagined_move_feats.reshape((n_allies, -1)), imagined_enemy_feats.reshape((n_allies, -1)), imagined_ally_feats.reshape((n_allies, -1)), imagined_own_feats], axis=1)
        imagined_obs = np.insert(imagined_other_obs, agent_id, obs, axis=0)
        return imagined_obs

    def imagined_obs_agent_sc2v2(self, agent_id, obs, n_enemies, n_allies, move_feats_dim, enemy_feats_dim, ally_feats_dim, own_feats_dim, sight_range, is_protoss, unit_type_bits):
        move_feats = obs[:move_feats_dim]
        enemy_feats = obs[move_feats_dim:move_feats_dim+n_enemies*enemy_feats_dim].reshape(n_enemies, enemy_feats_dim)
        ally_feats = obs[move_feats_dim+n_enemies*enemy_feats_dim:move_feats_dim+n_enemies*enemy_feats_dim+n_allies*ally_feats_dim].reshape(n_allies, ally_feats_dim)
        own_feats = obs[move_feats_dim+n_enemies*enemy_feats_dim+n_allies*ally_feats_dim:move_feats_dim+n_enemies*enemy_feats_dim+n_allies*ally_feats_dim+own_feats_dim]

        imagined_move_feats = np.zeros((n_allies, move_feats_dim))
        imagined_enemy_feats = np.zeros((n_allies, n_enemies, enemy_feats_dim))
        imagined_ally_feats = np.zeros((n_allies, n_allies, ally_feats_dim))
        imagined_own_feats = np.zeros((n_allies, own_feats_dim))

        for a_idx in range(n_allies):
            a_visible = ally_feats[a_idx][0]
            if a_visible == 0:
                continue
            # imagined movement features
            imagined_move_feats[a_idx] = move_feats # meaningless
            # imagined enemy features
            a_agent_id = a_idx if a_idx < agent_id else a_idx + 1
            a_sight_range = sight_range
            a_position = ally_feats[a_idx][2:4]
            for e_idx in range(n_enemies):
                e_visible = enemy_feats[e_idx][0]
                if e_visible == 0:
                    continue
                e_position = enemy_feats[e_idx][2:4]
                if math.dist(e_position, a_position) * sight_range <= a_sight_range:
                    imagined_enemy_feats[a_idx][e_idx] = enemy_feats[e_idx]
                    imagined_enemy_feats[a_idx][e_idx][0] = 1 # visible
                    imagined_enemy_feats[a_idx][e_idx][1] = math.dist(e_position, a_position) / sight_range # distance
                    imagined_enemy_feats[a_idx][e_idx][2] = (e_position[0] - a_position[0]) / sight_range # relative_x
                    imagined_enemy_feats[a_idx][e_idx][3] = (e_position[1] - a_position[1]) / sight_range # relative_y
            # imagined ally features
            for b_idx in range(n_allies):
                b_visible = ally_feats[b_idx][0]
                if b_visible == 0:
                    continue
                b_position = ally_feats[b_idx][2:4]
                if a_idx == b_idx:
                    imagined_ally_feats[a_idx][b_idx][0] = 1 # visible
                    imagined_ally_feats[a_idx][b_idx][1] = ally_feats[a_idx][1] # distance
                    imagined_ally_feats[a_idx][b_idx][2] = -ally_feats[a_idx][2] # relative_x
                    imagined_ally_feats[a_idx][b_idx][3] = -ally_feats[a_idx][3] # relative_y
                    ind = 0
                    imagined_ally_feats[a_idx][b_idx][ind+4] = own_feats[ind] # health
                    ind += 1
                    if is_protoss:
                        imagined_ally_feats[a_idx][b_idx][ind+4] = own_feats[ind] # shield
                        ind += 1
                    imagined_ally_feats[a_idx][b_idx][ind+4] = own_feats[ind] # stochatic attack
                    ind += 1
                    imagined_ally_feats[a_idx][b_idx][ind+4] = own_feats[ind] # stochastic health
                    ind += 1
                    if unit_type_bits > 0:
                        imagined_ally_feats[a_idx][b_idx][ind+4] = own_feats[ind] # unit_type
                    continue
                if math.dist(a_position, b_position) * sight_range <= a_sight_range:
                    imagined_ally_feats[a_idx][b_idx] = ally_feats[b_idx]
                    imagined_ally_feats[a_idx][b_idx][0] = 1 # visible
                    imagined_ally_feats[a_idx][b_idx][1] = math.dist(b_position, a_position) / sight_range # distance
                    imagined_ally_feats[a_idx][b_idx][2] = (b_position[0] - a_position[0]) / sight_range # relative_x
                    imagined_ally_feats[a_idx][b_idx][3] = (b_position[1] - a_position[1]) / sight_range # relative_y
            # imagined own features
            ind = 0
            imagined_own_feats[a_idx][ind] = ally_feats[a_idx][ind+4] # health
            ind += 1
            if is_protoss:
                imagined_own_feats[a_idx][ind] = ally_feats[a_idx][ind+4] # shield
                ind += 1
            imagined_own_feats[a_idx][ind] = ally_feats[a_idx][ind+4] # stochatic attack
            ind += 1
            imagined_own_feats[a_idx][ind] = ally_feats[a_idx][ind+4] # stochastic health
            ind += 1
            imagined_own_feats[a_idx][ind] = own_feats[ind] + ally_feats[a_idx][2] # x
            ind += 1
            imagined_own_feats[a_idx][ind] = own_feats[ind] + ally_feats[a_idx][3] # y
            ind += 1
            if unit_type_bits > 0:
                imagined_own_feats[a_idx][ind] = ally_feats[a_idx][-1] # unit_type

        imagined_other_obs = np.concatenate([imagined_move_feats.reshape((n_allies, -1)), imagined_enemy_feats.reshape((n_allies, -1)), imagined_ally_feats.reshape((n_allies, -1)), imagined_own_feats], axis=1)
        imagined_obs = np.insert(imagined_other_obs, agent_id, obs, axis=0)
        return imagined_obs
