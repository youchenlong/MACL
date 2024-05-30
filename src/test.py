import torch as th
from utils.contrastive import infoNCE_loss

consensus_features = th.rand(32, 10, 5, 16)
bs, ts, n_agents, consensus_dim = consensus_features.shape
consensus_features = consensus_features.view(-1, consensus_dim)
labels = th.arange(bs * ts, device=consensus_features.device).repeat_interleave(n_agents)
mask_positive = labels.unsqueeze(0) == labels.unsqueeze(1)
contrasTemperature = 0.5
contrastive_loss = infoNCE_loss(consensus_features, mask_positive, contrasTemperature)
print(contrastive_loss)