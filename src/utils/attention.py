import torch as th
import torch.nn as nn
import torch.nn.functional as F
from utils.sparsemax import Sparsemax

class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, n_dim, softTemperature, dim_q=None, dim_k=None, dim_v=None, verbose=False, isSoftmax=False):
        super(MultiHeadAttention, self).__init__()
        assert (n_dim % n_heads) == 0, "n_heads must divide n_dim"
        attn_dim = n_dim // n_heads
        self.attn_dim = attn_dim
        self.n_heads = n_heads
        self.verbose = verbose
        self.temperature=attn_dim ** 0.5 / softTemperature
        self.isSoftmax = isSoftmax
        if dim_q is None:
            dim_q = n_dim
        if dim_k is None:
            dim_k = dim_q
        if dim_v is None:
            dim_v = dim_k

        self.fc_q = nn.Linear(dim_q, n_dim, bias=False)
        self.fc_k = nn.Linear(dim_k, n_dim, bias=False)
        self.fc_v = nn.Linear(dim_v, n_dim)
        self.fc_final = nn.Linear(n_dim, n_dim)

    def forward(self, h_q, h_k, h_v, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        bs = h_q.shape[0]
        q = self.fc_q(h_q).view(bs, -1, self.n_heads, self.attn_dim).transpose(1, 2)
        k_T = self.fc_k(h_k).view(bs, -1, self.n_heads, self.attn_dim).permute(0, 2, 3, 1)
        v = self.fc_v(h_v).view(bs, -1, self.n_heads, self.attn_dim).transpose(1, 2)
        alpha = th.matmul(q / self.temperature, k_T)
        if self.isSoftmax:
            alpha = F.softmax(alpha, dim=-1)
        else:
            sparsemax = Sparsemax(dim=-1)
            alpha = sparsemax(alpha)
        if self.verbose:
            assert self.n_heads == 1
            self.alpha = alpha.squeeze(2).detach()
        res = th.matmul(alpha, v).transpose(1, 2).reshape(bs, -1, self.attn_dim * self.n_heads)
        res = self.fc_final(res)
        return res