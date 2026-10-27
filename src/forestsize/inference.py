import time
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from forestsize.simulation import SumStat
from forestsize.parameters import SAMPLE_SIZE







class SimDataset(Dataset):
    def __init__(self, sim_ids):
        self.sim_ids = sim_ids
    def __len__(self):
        return len(self.sim_ids)
    def __getitem__(self, index):
        sim_id = self.sim_ids[index]
        
        theta, _, _, SAD, LTT, seeds = SumStat(sim_id)
        
        SAD = torch.from_numpy(np.log1p(SAD))
        LTT = torch.from_numpy(LTT)
        theta = torch.from_numpy(np.log10(theta))
        
        return theta, SAD, LTT




class EcoEvoModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
        self.fc1 = torch.nn.Linear(SAMPLE_SIZE, 64)
        self.act = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(64, 4)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x
    


def benchmark():
    n_sim = 100
    n_workers = 0
    batch_size = None
    
    dataset = SimDataset(range(n_sim))
    loader = DataLoader(
        dataset,
        batch_size = batch_size,
        num_workers = n_workers
        )

    start = time.perf_counter()
    
    for sim in loader:
        pass
    
    end = time.perf_counter()
    
    elapsed = end - start
    throughput = n_sim / elapsed
    print(f'Simulations : {n_sim}')
    print(f'in a total time of : {elapsed:.2f} s')
    print(f'\nThroughput : {throughput : .2f} sim/s')
    

def ModelTraining():
    
    n_train = 100
    n_valid = 20
    n_test = 20

    n_workers = 4
    batch_size = 8
    n_epochs = 5
    
    valid_ids = range(1000000, 1000000 + n_valid)
    test_ids = range(1000000 + n_valid, 1000000 + n_valid + n_test) #Attention temporaire, si n_train = 1000000 on overlap
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")
    

    valid_dataset = SimDataset(valid_ids)
    test_dataset = SimDataset(test_ids)
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        num_workers=n_workers
        )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        num_workers=n_workers
        )
    
    net = EcoEvoModel().to(device)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    
    for epoch in range(n_epochs):
        
        train_ids = range(epoch * n_train, (epoch+1) * n_train)
        train_dataset = SimDataset(train_ids)
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            num_workers=n_workers,
            shuffle=True
            )
        
        net.train()
        
        total_train_loss = 0.0
        n_train_samples = 0
    
        for params, sad in train_loader:
            
            params = params.to(device)
            sad = sad.to(device)
            
            optimizer.zero_grad()
            
            prediction = net(sad)
            loss = criterion(prediction, params)
            
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item() * sad.size(0)
            n_train_samples += sad.size(0)
            
        train_loss = total_train_loss / n_train_samples
        
        net.eval()
        total_valid_loss = 0.0
        n_valid_samples = 0
        
        with torch.no_grad():
            
            for params, sad in valid_loader:
            
                params = params.to(device)
                sad = sad.to(device)
                
                prediction = net(sad)
                loss = criterion(prediction, params)
                
                total_valid_loss += loss.item() * sad.size(0)
                n_valid_samples += sad.size(0)
                
            valid_loss = total_valid_loss / n_valid_samples
            
            print(f'Epoch {epoch + 1} | ' +
                  f'Train loss : {train_loss:.4f} | ' +
                  f'Valid loss : {valid_loss:.4f}')
    
    net.eval()
    
    total_test_loss = 0.0
    n_test_samples = 0
    
    with torch.no_grad():
        
        for params, sad in test_loader:
            
            params = params.to(device)
            sad = sad.to(device)
            
            prediction = net(sad)
            loss = criterion(prediction, params)
            
            total_test_loss += loss.item() * sad.size(0)
            n_test_samples += sad.size(0)
            
        test_loss = total_test_loss / n_test_samples
        
        print(f'Test loss : {test_loss:.4f}')