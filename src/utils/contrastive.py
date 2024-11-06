import torch
import torch.nn.functional as F

# def contrastive_loss(embedding, temperature=1.0):
#     embedding = F.normalize(embedding, p=2, dim=-1)
#     batch_size, timesteps, n_agents, embedding_dim = embedding.size()
#     sim_matrix = torch.matmul(embedding.view(batch_size, -1, embedding_dim), embedding.view(batch_size, -1, embedding_dim).transpose(1, 2)) / temperature
#     sim_matrix = sim_matrix - torch.max(sim_matrix, dim=2, keepdim=True)[0]
#     positive_sim = []
#     for t in range(timesteps):
#         for i in range(n_agents):
#             for j in range(n_agents):
#                 positive_sim.append(sim_matrix[:, t*n_agents+i, t*n_agents+j])
#                 positive_sim.append(sim_matrix[:, t*n_agents+j, t*n_agents+i])
#     positive_sim = torch.stack(positive_sim, dim=1) # [bs, timesteps*n_agents*n_agents*2]

#     negative_sim = []
#     for t1 in range(timesteps):
#         for t2 in range(timesteps):
#             for i in range(n_agents):
#                 for j in range(n_agents):
#                     negative_sim.append(sim_matrix[:, t1*n_agents+i, t2*n_agents+j])
#                     negative_sim.append(sim_matrix[:, t2*n_agents+j, t1*n_agents+i])
#     negative_sim = torch.stack(negative_sim, dim=1) # [bs, timesteps*timesteps*n_agents*n_agents*2]

#     pos_exp = torch.exp(positive_sim)
#     neg_exp = torch.exp(negative_sim)
#     loss = -torch.mean(torch.log(torch.sum(pos_exp, dim=1) / torch.sum(neg_exp, dim=1)))

#     return loss

def contrastive_loss(embedding, temperature=1.0):
    embedding = F.normalize(embedding, p=2, dim=-1)
    batch_size, timesteps, n_agents, embedding_dim = embedding.size()
    sim_matrix = torch.matmul(embedding.view(batch_size, -1, embedding_dim), embedding.view(batch_size, -1, embedding_dim).transpose(1, 2)) / temperature
    sim_matrix = sim_matrix - torch.max(sim_matrix, dim=2, keepdim=True)[0]

    positive_sim = sim_matrix.view(batch_size, timesteps, n_agents, timesteps, n_agents)
    positive_sim = positive_sim[:, range(timesteps), :, range(timesteps), :].contiguous().view(batch_size, -1)

    negative_sim = sim_matrix.view(batch_size, timesteps, n_agents, timesteps, n_agents)
    negative_sim = negative_sim.view(batch_size, -1)

    pos_exp = torch.exp(positive_sim)
    neg_exp = torch.exp(negative_sim)
    loss = -torch.mean(torch.log(torch.sum(pos_exp, dim=1) / torch.sum(neg_exp, dim=1)))

    return loss



if __name__ == '__main__':
    batch_size = 32
    timesteps = 100
    n_agents = 10
    embedding_dim = 64
    embedding = torch.randn(batch_size, timesteps, n_agents, embedding_dim)
    loss = contrastive_loss(embedding)
    print(loss)