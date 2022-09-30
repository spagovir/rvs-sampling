from dataclasses import dataclass
import torch as t
import torch.nn as nn
from dataclasses import dataclass
from torch.utils.data import DataLoader, Dataset, TensorDataset

""" Three separate MLPs for predicting P(A|R,S), P(A|S), and P(R|S)."""

@dataclass
class MLPConfig:
    num_reward_bins : int
    num_state_dims : int 
    num_actions : int
    device : str
    epochs : int
    lr : float = 0.0001
    intermediate_dim : int = 1024
    num_hidden_layers : int = 2
    batch_size = 16

class RvSMLP(nn.Module):
    def __init__(self, config : MLPConfig):
        super().__init__()
        self.reward_embed = nn.Embedding(config.num_reward_bins, config.intermediate_dim)
        self.state_embed = nn.Linear(config.num_state_dims, config.intermediate_dim)
        layers = []
        for i in range(config.num_hidden_layers):
            layers.append(nn.GELU())
            layers.append(nn.Linear(config.intermediate_dim, config.num_actions if i == config.num_hidden_layers - 1 else config.intermediate_dim))
        self.layers = nn.ModuleList(layers)

    def forward(self, reward : int, state : t.Tensor):
        x = self.reward_embed(reward) + self.state_embed(state)
        for layer in self.layers:
            x = layer(x)
        return x
''' net for predicting unconditional actions'''
class AMLP(nn.Module):
    def __init__(self, config : MLPConfig):
        super().__init__()
        self.state_embed = nn.Linear(config.num_state_dims, config.intermediate_dim)
        layers = []
        for i in range(config.num_hidden_layers):
            layers.append(nn.GELU())
            layers.append(nn.Linear(config.intermediate_dim, config.num_actions if i == config.num_hidden_layers - 1 else config.intermediate_dim))
        self.layers = nn.ModuleList(layers)
    def forward(self, state : t.Tensor):
        x = self.state_embed(state)
        for layer in self.layers:
            x = layer(x)
        return x
''' net for predicting unconditional rewards'''
class RMLP(nn.Module):
    def __init__(self, config : MLPConfig):
        super().__init__()
        self.state_embed = nn.Linear(config.num_state_dims, config.intermediate_dim)
        layers = []
        for i in range(config.num_hidden_layers):
            layers.append(nn.GELU())
            layers.append(nn.Linear(config.intermediate_dim, config.num_reward_bins if i == config.num_hidden_layers - 1 else config.intermediate_dim))
        self.layers = nn.ModuleList(layers)
    def forward(self, state : t.Tensor):
        x = self.state_embed(state)
        for layer in self.layers:
            x = layer(x)
        return x

def train(config : MLPConfig, dataset : Dataset, mainNet : RvSMLP, actionNet : AMLP, rewardNet : RMLP):
    dl = DataLoader(dataset, MLPConfig.batch_size, True)
    mainOptim = t.optim.Adam(mainNet.parameters(), config.lr)
    actionOptim = t.optim.Adam(actionNet.parameters(),config.lr)
    rewardOptim = t.optim.Adam(rewardNet.parameters(), config.lr)
    loss = nn.CrossEntropyLoss()
    for _ in range(config.epochs):
        for r, s, a in dl:
            aPredCond = mainNet(r,s)
            aPredUnCond = actionNet(s)
            rPredUnCond = rewardNet(s)
            aPredCondLoss = loss(aPredCond, a)
            aPredUnCondLoss = loss(aPredUnCond, a)
            rPredUnCondLoss = loss(rPredUnCond, r)
            mainOptim.zero_grad()
            actionOptim.zero_grad()
            rewardOptim.zero_grad()
            aPredCondLoss.backward()
            aPredUnCondLoss.backward()
            rPredUnCondLoss.backward()
            mainOptim.step()
            actionOptim.step()
            rewardOptim.step()
