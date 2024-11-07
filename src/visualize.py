import json
import os
import numpy as np
from sklearn import preprocessing
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def load_data(filename):
    with open(filename, 'r') as f:
        raw_dict = json.load(f)
        data = np.array(raw_dict['hidden_states']['values'])
        label = np.array(raw_dict['hidden_states']['steps'])
    timesteps, _, n_agents, rnn_hidden_dim = data.shape
    data = data.reshape(timesteps * n_agents, rnn_hidden_dim)
    label = label.repeat(n_agents)
    return data, label

def process(data):
    trans = TSNE(perplexity=5, n_components=2, n_iter=500, random_state=33)

    data = trans.fit_transform(data)
    return data

def visualize(datas):
    print(datas['data'].shape)
    print(datas['label'].shape)
    
    data_tsne = datas['data']
    label = datas['label']

    plt.figure(figsize=(10, 10))
    plt.scatter(data_tsne[:, 0], data_tsne[:, 1], c=label, marker='o', cmap='viridis')
    plt.show()

data, label = load_data('/home/oseasy/桌面/MARL/MACL/results/sacred/macl/lbf/6/metrics.json')
data_tsne = process(data)
datas = {'data':data_tsne, 'label':label}
visualize(datas)
