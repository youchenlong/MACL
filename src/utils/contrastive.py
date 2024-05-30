import torch as th

def infoNCE_loss(features, positive_mask, contrasTemperature):
    """
    features: [bs, feature_dim]
    positive_mask: [bs, bs]
    \begin{equation*}
        \begin{aligned}
            -\log\frac{exp(g(q, k_+))}{\sum_{i=1}^{k}exp(g(q, k_i))}
        \end{aligned}
    \end{equation*}
    """
    # similarity matrix
    sim_matrix = th.matmul(features, features.T) / contrasTemperature
    # Stability trick: Subtract the maximum value for numerical stability
    logits_max, _ = th.max(sim_matrix, dim=1, keepdim=True)
    logits = sim_matrix - logits_max.detach()
    # negative logits
    exp_logits = th.exp(logits)
    exp_sum = exp_logits.sum(1, keepdim=True) - exp_logits * positive_mask
    # positive logits
    log_prob = logits - th.log(exp_sum)
    # mask out the negative samples
    log_prob_pos = (positive_mask * log_prob).sum(1) / positive_mask.sum(1)
    # contrastive loss
    contrastive_loss = -log_prob_pos.mean()
    return contrastive_loss