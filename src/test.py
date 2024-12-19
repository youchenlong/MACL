import os
import time
import numpy as np
import math
import yaml

from envs import register_smac, register_smacv2, REGISTRY

def imagined_obs_agent_sc2(agent_id, obs, n_enemies, n_allies, move_feats_dim, enemy_feats_dim, ally_feats_dim, own_feats_dim, sight_range, is_protoss, unit_type_bits):
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
                if is_protoss > 0:
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


def imagined_obs_agent_sc2v2(agent_id, obs, n_enemies, n_allies, move_feats_dim, enemy_feats_dim, ally_feats_dim, own_feats_dim, sight_range, is_protoss, unit_type_bits):
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

    imagined_other_obs = np.concatenate([imagined_move_feats.reshape((n_allies, -1)), imagined_enemy_feats.reshape((n_allies, -1)), imagined_ally_feats.reshape((n_allies, -1)), imagined_own_feats], axis=1) # [n_allies, n_features]
    imagined_obs = np.insert(imagined_other_obs, agent_id, obs, axis=0) # [n_agents, n_features]
    return imagined_obs

if __name__ == "__main__":

    # seed = 0
    # env_name = "sc2"
    # with open(os.path.join(os.path.dirname(__file__), "config", "envs", "{}.yaml".format(env_name)), "r") as f:
    #     config_dict = yaml.load(f, Loader=yaml.FullLoader)
    #     env_config = config_dict["env_args"]
    # register_smac()
    # env_config["map_name"] = "2s_vs_1sc"
    # env_config["seed"] = seed
    # env = REGISTRY[env_name](**env_config)
    # env.reset()
    # nf_move = env.env.get_obs_move_feats_size()
    # (n_enemies, nf_en) = env.env.get_obs_enemy_feats_size()
    # (n_allies, nf_al) = env.env.get_obs_ally_feats_size()
    # nf_own = env.env.get_obs_own_feats_size()
    # print(nf_move, nf_own, nf_al, nf_en, n_allies, n_enemies)
    # sight_range = 9
    # is_protoss = True
    # unit_type_bits = 0
    # agent_id = 0
    # obs = env.get_obs_agent(agent_id)
    # imagined_obs = imagined_obs_agent_sc2(agent_id, obs, n_enemies, n_allies, nf_move, nf_en, nf_al, nf_own, sight_range, is_protoss, unit_type_bits)
    # # print(imagined_obs)
    # time.sleep(100)

    
    
    seed = 0
    env_name = "sc2v2"
    with open(os.path.join(os.path.dirname(__file__), "config", "envs", "{}.yaml".format(env_name)), "r") as f:
        config_dict = yaml.load(f, Loader=yaml.FullLoader)
        env_config = config_dict["env_args"]
    register_smacv2()
    env_config["map_name"] = "zerg_10_vs_10"
    env_config["seed"] = seed
    env = REGISTRY[env_name](**env_config)
    env.reset()
    nf_move = env.env.get_obs_move_feats_size()
    (n_enemies, nf_en) = env.env.get_obs_enemy_feats_size()
    (n_allies, nf_al) = env.env.get_obs_ally_feats_size()
    nf_own = env.env.get_obs_own_feats_size()
    print(nf_move, nf_own, nf_al, nf_en, n_allies, n_enemies)
    sight_range = 9
    is_protoss = True
    unit_type_bits = 3
    agent_id = 0
    obs = env.get_obs_agent(agent_id)
    imagined_obs = imagined_obs_agent_sc2v2(agent_id, obs, n_enemies, n_allies, move_feats_dim, enemy_feats_dim, ally_feats_dim, own_feats_dim, sight_range, is_protoss, unit_type_bits)
    print(imagined_obs.shape)
    time.sleep(100)

    print("hello")