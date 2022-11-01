from argparse import Action
from dataclasses import dataclass
from typing import List
from tqdm import tqdm
import torch as t
import torch.nn as nn
from dataclasses import dataclass
from torch.utils.data import DataLoader, Dataset, TensorDataset
from rtg_replay_buffer import RtGBuffer
from abc import ABC, abstractclassmethod
import wandb
from einops import rearrange

""" Three separate MLPs for predicting P(A|R,S), P(A|S), and P(R|S)."""

@dataclass
class MLPConfig:
    num_reward_bins : int
    num_state_dims : int 
    num_actions : int
    device : str
    epochs : int
    batches_per_buffer : int
    lr : float = 0.001
    intermediate_dim : int = 2048
    num_hidden_layers : int = 2
    batch_size = 256
    save_every = 10000
    log_every = 4
    reward_weight = 1

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
        self.reward_weight = config.reward_weight

    def forward(self, reward : int, state : t.Tensor):
        x = self.reward_embed(reward) * config.reward_weight + self.state_embed(state)
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

class Binner(nn.Module):
    def __init__(self, config : MLPConfig):
        super().__init__()
        self.register_buffer('running_min_max', t.tensor([0.0, 1.0]))
        self.num_bins = config.num_reward_bins
    def forward(self, x):
        self.running_min_max[0] = min(self.running_min_max[0], t.min(x).item())
        self.running_min_max[1] = max(self.running_min_max[1], t.max(x).item())
        return t.floor((x - self.running_min_max[0])/(self.running_min_max[1]-self.running_min_max[0] + 0.0001) * self.num_bins).long()
    def bin_value(self, n : int) -> float:
        return self.running_min_max[0] + (self.running_min_max[1] - self.running_min_max[0]) / self.num_bins * n

class RvSModel(nn.Module):
    def __init__(self, config : MLPConfig):
        super().__init__()
        self.mainNet = RvSMLP(config)
        self.actionNet = AMLP(config)
        self.rewardNet = RMLP(config)
        self.binner = Binner(config)
    def forward(self, r, s):
        binnedR = self.binner(r)
        aPredCond = self.mainNet(binnedR, s)
        aPredUnCond = self.actionNet(s)
        rPredUnCond = self.rewardNet(s)
        return (binnedR, aPredCond, aPredUnCond,rPredUnCond)

class TrainLogger:
    def __init__(self, name):
        self.name = name
        wandb.init(project=name)
        self.chkpt = 0
    def log(self, i, aPredCondLoss, aPredUnCondLoss, rPredUnCondLoss, running_min_max):
        wandb.log(
        { "step": i 
        , "conditioned action loss": aPredCondLoss
        , "unconditioned action loss": aPredUnCondLoss
        , "unconditioned reward loss": rPredUnCondLoss
        , "running min": running_min_max[0]
        , "running max": running_min_max[1]})
    def save(self, state_dict):
        t.save(state_dict, f"./checkpoints/{self.name}-{self.chkpt}")
        wandb.save(f"./checkpoints/{self.name}-{self.chkpt}")
        self.chkpt += 1


    


def train(config : MLPConfig, dataset : Dataset, model : RvSModel, saveDir : str):
    dl = DataLoader(dataset, MLPConfig.batch_size, True)
    model.to(config.device)
    mainOptim = t.optim.Adam(model.mainNet.parameters(), config.lr)
    actionOptim = t.optim.Adam(model.actionNet.parameters(),config.lr)
    rewardOptim = t.optim.Adam(model.rewardNet.parameters(), config.lr)
    loss = nn.CrossEntropyLoss()
    for _ in tqdm(range(config.num_steps)):
        for r, s, a in dl:
            r = r.to(config.device)
            s = s.to(config.device)
            a = a.to(config.device)
            binnedR, aPredCond, aPredUnCond, rPredUnCond = model(r,s)
            aPredCondLoss = loss(aPredCond, a)
            aPredUnCondLoss = loss(aPredUnCond, a)
            rPredUnCondLoss = loss(rPredUnCond, binnedR)
            mainOptim.zero_grad()
            actionOptim.zero_grad()
            rewardOptim.zero_grad()
            aPredCondLoss.backward()
            aPredUnCondLoss.backward()
            rPredUnCondLoss.backward()
            mainOptim.step()
            actionOptim.step()
            rewardOptim.step()


def trainWDopamine(config : MLPConfig, game : str, model : RvSModel, logger : TrainLogger):
    buffers = [
        RtGBuffer(
            observation_shape=(84,84), 
            stack_size = 4, 
            replay_capacity = 1000000, 
            batch_size = config.batch_size,
            update_horizon=4000,
            gamma = 0.99999)
        for i in range(10)
    ]
    for i in tqdm(range(10)):
        buffers[i].load(f"./atari_data/dqn/{game}/1/replay_logs/", 5 * i)
    model.to(config.device)
    mainOptim = t.optim.Adam(model.mainNet.parameters(), config.lr)
    actionOptim = t.optim.Adam(model.actionNet.parameters(),config.lr)
    rewardOptim = t.optim.Adam(model.rewardNet.parameters(), config.lr)
    loss = nn.CrossEntropyLoss()
    steps = 0
    for epoch in tqdm(range(config.epochs)):
        buffer_idx = t.randint(10, (1,)).item()
        # for buffer_idx in buffer_idxs:
            # buffer = RtGBuffer(
                # observation_shape=(84,84), 
                # stack_size = 4, 
                # replay_capacity = 1000000, 
                # batch_size = 32,
                # update_horizon=200)
            # buffer.load(f"./atari_data/dqn/{game}/1/replay_logs/", buffer_idx)
        # for _ in range(config.batches_per_buffer): 
        s, a, r, _, _, _, _, _ = buffers[buffer_idx].sample_transition_batch()
        r = t.from_numpy(r).to(config.device)
        s = t.from_numpy(s).to(config.device)
        a = t.from_numpy(a).to(config.device).long()
        s = rearrange(s, "b h w s -> b (h w s)").float() / 256
        binnedR, aPredCond, aPredUnCond, rPredUnCond = model(r,s)
        aPredCondLoss = loss(aPredCond, a)
        aPredUnCondLoss = loss(aPredUnCond, a)
        rPredUnCondLoss = loss(rPredUnCond, binnedR)
        mainOptim.zero_grad()
        actionOptim.zero_grad()
        rewardOptim.zero_grad()
        aPredCondLoss.backward()
        aPredUnCondLoss.backward()
        rPredUnCondLoss.backward()
        mainOptim.step()
        actionOptim.step()
        rewardOptim.step()
        steps += config.batch_size
        if epoch % config.save_every == 0:
            logger.save(model.state_dict())
        if epoch % config.log_every == 0:
            logger.log(steps, aPredCondLoss, aPredUnCondLoss, rPredUnCondLoss, model.binner.running_min_max)

# %%
config = MLPConfig(32, 84*84*4, 18, 'cuda', 500000, 1000)
model = RvSModel(config)
logger = TrainLogger("SpaceInvadersRun6")
game = "SpaceInvaders"
trainWDopamine(config, game, model, logger)