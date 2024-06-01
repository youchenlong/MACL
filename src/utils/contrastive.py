import torch as th

def infoNCE_loss(features, positive_mask, contrasTemperature, batch_size=512):
    """
    features: [bs, feature_dim]
    positive_mask: [bs, bs]
    \begin{equation*}
        \begin{aligned}
            -\log\frac{exp(g(q, k_+))}{\sum_{i=1}^{k}exp(g(q, k_i))}
        \end{aligned}
    \end{equation*}
    """
    bs, feature_dim = features.size()
    total_loss = 0
    n_batchs = (bs + batch_size - 1) // batch_size
    for i in range(n_batchs):
        start = i * batch_size
        end = min(start + batch_size, bs)
        features_batch = features[start:end]
        positive_mask_batch = positive_mask[start:end]
        # similarity matrix
        sim_matrix = th.matmul(features_batch, features.T) / contrasTemperature
        # Stability trick: Subtract the maximum value for numerical stability
        logits_max, _ = th.max(sim_matrix, dim=1, keepdim=True)
        logits = sim_matrix - logits_max.detach()
        # negative logits
        exp_logits = th.exp(logits)
        exp_sum = exp_logits.sum(1, keepdim=True) - exp_logits * positive_mask_batch
        # positive logits
        log_prob = logits - th.log(exp_sum)
        # mask out the negative samples
        log_prob_pos = (positive_mask_batch * log_prob).sum(1) / positive_mask_batch.sum(1)
        total_loss += -log_prob_pos.mean()
    total_loss /= n_batchs
    return total_loss