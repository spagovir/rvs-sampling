# %%
from rvs import MLPConfig, RvSMLP, AMLP, RMLP, train
import torch as t
import torch.nn as nn
from torch.utils.data import Dataset, TensorDataset 
from typing import Tuple

def test1() -> Tuple[Dataset, MLPConfig]:
    config = MLPConfig(2, 1, 2, 'cpu', 1000)
    states = t.rand((1000,1))
    actions = t.randint(2,(1000,))
    rewards = t.logical_xor(states.squeeze(1) > 0.5, actions).long()
    return (TensorDataset(rewards, states, actions), config)
# %%

def runTest1():
    ds, config = test1()
    mainNet = RvSMLP(config)
    actionNet = AMLP(config)
    rewardNet = RMLP(config)
    train(config, ds, mainNet, actionNet, rewardNet)
    return (mainNet, actionNet, rewardNet)

# %%
# %%
