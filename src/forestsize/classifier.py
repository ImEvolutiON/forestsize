import torch
from torch import nn
from torch.utils.data import Dataset

from forestsize.simulation import SumStat
from forestsize.parameters import H0, H1, CHANGETIMES

def classifier_sim(sim_id):
    
    hypothesis = sim_id % 2
    
    theta, G, T, SAD, LTT, hypothesis, seeds = SumStat(sim_id)
    
    if G is None:
        return None
    
    return hypothesis, theta, G, T, SAD, LTT

def borne_sup(G):
    
    return prediction, logBF

class GNN_Encoder(nn.Module):
    pass

class LTT_Encoder(nn.Module):
    pass

class SAD_Encoder(nn.Module):
    pass

class ClassifierHead(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.fc1 = nn.Linear(64, 32)
        self.act = nn.ELU()
        self.fc2 = nn.Linear(32, 1)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x
    

class Classifier(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        
        self.encoder = encoder
        self.head = ClassifierHead()
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.head(x)
        return x