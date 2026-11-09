import torch
import numpy as np

from forestsize.parameters import SAMPLE_SIZE

class EncoderSAD(torch.nn.Module):
    """
    The Species Abundances Distribution can be seen as a frequency spectrum
    especially when accuracy can reach 50000 different bins in exact definition
    where each bin is incremented by +1 abundance.
    
    See Tran et al. (2024)
    
    log1p (in modeltraining) --> Nekola et al 2008 Artifications in the Log-transofrmation of Species Abundance Distributions 
    Ne pas normaliser car on perd l'information de la richesse 
    """
    
    def __init__(self):
        super().__init__()
        
        self.fc1 = torch.nn.Linear(SAMPLE_SIZE, 64)
        self.fc2 = torch.nn.Linear(64, 64)
        self.fc3 = torch.nn.Linear(64, 64)
        
        self.act = torch.nn.ReLU()
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.act(x)
        x = self.fc3(x)
        return x


n_layer = 3
ker_size = 5
n_hidden = 8

class EncoderLTT(torch.nn.Module):
    """
    LTT source
    Because LTT may have different length, we need to
    log1p because coalescent may have very deep time 10e15 (in modeltraining)
    
    Lajaati 2023
    """
    
    def __init__(self):
        super().__init__()
        
        self.conv1 = torch.nn.Conv1d(1, 8, kernel_size=5)
        self.conv2 = torch.nn.Conv1d(8, 16, kernel_size=5)
        self.conv3 = torch.nn.Conv1d(16, 32, kernel_size=5)
        
        # Compute output length
        length = SAMPLE_SIZE
        for _ in range(n_layer):
            length = (length - ker_size+1) // 2
        
        self.fc1 = torch.nn.Linear(4*n_hidden*length, 100)
        self.fc2 = torch.nn.Linear(100, 64)
        
        self.act = torch.nn.ReLU()
        self.pool = torch.nn.AvgPool1d(kernel_size=2)
        self.dropout = torch.nn.Dropout(0.01)
        
        
    def forward(self, x):
        # First conv
        x = self.conv1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.pool(x)
        
        # Second conv
        x = self.conv2(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.pool(x)
        
        # Third conv
        x = self.conv3(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.pool(x)
        
        # FC layers
        x = torch.flatten(x, start_dim=1) # regroup all conv dim
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class EncoderT(torch.nn.Module):
    pass

class EncoderG(torch.nn.Module):
    # Ici soit un NN pour apprendre G, soit directement l'oracle distribution des temps de coalescence
    pass

