import time
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from utils.contrastive import ContrastiveLoss

bs = 128
ts = 200
n_agents = 8
consensus_dim = 4

loss_func = ContrastiveLoss(ts)
loss = th.tensor(0.0).cuda()

start_time = time.time()
prediction = th.rand(bs, ts, n_agents * consensus_dim).cuda()
target_projection = th.rand(bs, ts, n_agents * consensus_dim).cuda()
for emb_i, emb_j in zip(prediction, target_projection.detach()):
    loss += loss_func(emb_i, emb_j)
end_time = time.time()
print("cost time: ", end_time - start_time)
print(loss/(bs*ts))

start_time = time.time()
emb_i = th.zeros(ts, n_agents * consensus_dim).cuda()
emb_j = th.zeros(ts, n_agents * consensus_dim).cuda()
loss = loss_func(emb_i, emb_j)
end_time = time.time()
print("cost time: ", end_time - start_time)
print(loss)