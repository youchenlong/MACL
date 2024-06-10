import torch as th
import torch.nn as nn
import torch.nn.functional as F

class ContrastiveLoss(nn.Module):
    def __init__(self, batch_size, device='cuda', temperature=0.1):
        super(ContrastiveLoss, self).__init__()
        self.batch_size = batch_size
        self.device = device
        self.temperature = th.tensor(temperature, device=device)
        self.negative_mask = (~th.eye(batch_size * 2, batch_size * 2, dtype=bool, device=device)).float()

    def forward(self, emb_i, emb_j):
        """
        emb_i: [bs, feature_dim]
        emb_j: [bs, feature_dim]
        """     
        # z_i = F.normalize(emb_i, dim=1)
        # z_j = F.normalize(emb_j, dim=1)
        z_i = emb_i
        z_j = emb_j
        representations = th.cat([z_i, z_j], dim=0) # [2 * bs, feature_dim]
        # similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2) # [2 * bs, 2 * bs]
        similarity_matrix = F.mse_loss(representations.unsqueeze(1), representations.unsqueeze(0), reduction='none').mean(dim=-1) # [2 * bs, 2 * bs]
        sim_ij = th.diag(similarity_matrix, self.batch_size) # [bs]
        sim_ji = th.diag(similarity_matrix, -self.batch_size) # [bs]
        positives = th.cat([sim_ij, sim_ji], dim=0) # [2 * bs]

        # nominator = th.exp(positives / self.temperature) # [2 * bs]
        # denominator = th.exp(similarity_matrix / self.temperature) * self.negative_mask # [2 * bs, 2 * bs]
        # loss_partial = -th.log(nominator / th.sum(denominator, dim=1)) # [2 * bs]

        # TODO: mask out the diagonal
        loss_partial = -th.log(th.exp(positives / self.temperature)) + th.logsumexp(similarity_matrix / self.temperature, dim=1) # [2 * bs]
        loss = th.sum(loss_partial) / (2 * self.batch_size)
        return loss