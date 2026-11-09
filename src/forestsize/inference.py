import time
import os
import csv
from itertools import product
import numpy as np
import torch
from sbi.neural_nets import posterior_nn
from torch.utils.data import Dataset, DataLoader
from forestsize.simulation import SumStat
from forestsize.parameters import SAMPLE_SIZE
from forestsize.encoders import EncoderSAD, EncoderLTT



class SimDataset(Dataset):
    def __init__(self, sim_ids):
        self.sim_ids = sim_ids
    def __len__(self):
        return len(self.sim_ids)
    def __getitem__(self, index):
        sim_id = self.sim_ids[index]
        
        theta, G, T, SAD, LTT, seeds = SumStat(sim_id)
                
        if any(level is None for level in (theta, G, T, SAD, LTT)):
            raise RuntimeError(f"Incomplete simulation: sim_id={sim_id}")
                
        return theta, G, T, SAD, LTT


class EcoEvoModel(torch.nn.Module):
    """
    Joint training of an embedding network and a NPE
    """
    
    def __init__(self, embedding_net, normalizing_flow):
        super().__init__()
        self.embedding_net = embedding_net       # Automatically registered encoder as submodule
        self.normalizing_flow = normalizing_flow # Automatically registered NPE as submodule
        
    def loss(self, theta, features):
        embedding = self.embedding_net(features)                   # Compute embeddings from the features of the current batch
        loss = self.normalizing_flow.loss(theta, embedding).mean() # Mean NPE loss over the batch
        return loss
    

def ram_used(pid):
    try:
        with open(f"/proc/{pid}/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024
    except (FileNotFoundError, ProcessLookupError):
        pass

    return 0.0


def total_ram():
    pid = os.getpid()

    with open(f"/proc/{pid}/task/{pid}/children") as f:
        workers = [int(p) for p in f.read().split()]

    return ram_used(pid) + sum(ram_used(p) for p in workers)

def benchmark():

    # Fixed simulation budget
    n_train = 32
    n_val = 8
    n_epochs = 1

    # Parameters to benchmark
    batch_sizes = [8, 16, 32]
    worker_counts = [2, 4]
    prefetch_factors = [1, 2]

    configurations = list(product(
        batch_sizes,
        worker_counts,
        prefetch_factors
    ))

    results = []

    for i, (batch_size, n_workers, prefetch_factor) in enumerate(configurations, 1):

        print(
            f"\nConfiguration {i}/{len(configurations)} | "
            f"Batch: {batch_size} | "
            f"Workers: {n_workers} | "
            f"Prefetch: {prefetch_factor}",
            flush=True
        )

        # Same neural network initialization for each configuration
        torch.manual_seed(123456)

        # Reset GPU memory statistics
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()

        start = time.perf_counter()

        # Training and validation
        wait_times, train_times, observed_ram = ModelTraining(
            n_train=n_train,
            n_val=n_val,
            n_epochs=n_epochs,
            n_workers=n_workers,
            batch_size=batch_size,
            prefetch_factor=prefetch_factor
        )

        # GPU measurements
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            peak_vram = torch.cuda.max_memory_allocated() / 1024**2
        else:
            peak_vram = 0.0

        elapsed = time.perf_counter() - start

        # Total simulation throughput (training + validation)
        n_simulations = (n_train + n_val) * n_epochs
        throughput = n_simulations / elapsed

        # Store measurements
        results.append({
            "batch_size": batch_size,
            "n_workers": n_workers,
            "prefetch_factor": prefetch_factor,
            "elapsed_s": elapsed,
            "simulations_per_second": throughput,
            "observed_ram_mib": observed_ram,
            "peak_vram_mib": peak_vram,
            "mean_wait_s": float(np.mean(wait_times)),
            "mean_train_s": float(np.mean(train_times))
        })

    # Save all measurements
    with open("benchmark_results.csv", "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    # Sort configurations by execution time
    ranked = sorted(results, key=lambda x: x["elapsed_s"])

    # Generate final report
    lines = [
        "========== FORESTSIZE BENCHMARK ==========",
        f"Configurations tested: {len(configurations)}",
        f"Training simulations per configuration: {n_train}",
        f"Validation simulations per configuration: {n_val}",
        ""
    ]

    for rank, result in enumerate(ranked, 1):
        lines.append(
            f"{rank}. "
            f"Batch={result['batch_size']} | "
            f"Workers={result['n_workers']} | "
            f"Prefetch={result['prefetch_factor']} | "
            f"Time={result['elapsed_s']:.2f}s | "
            f"Throughput={result['simulations_per_second']:.3f} sim/s | "
            f"RAM={result['observed_ram_mib']:.2f} MiB | "
            f"VRAM={result['peak_vram_mib']:.2f} MiB | "
            f"Wait={result['mean_wait_s']:.2f}s | "
            f"Train={result['mean_train_s']:.2f}s"
        )

    report = "\n".join(lines)

    print("\n" + report)

    with open("benchmark_report.txt", "w") as file:
        file.write(report + "\n")

    return results
    

def ModelTraining(n_train=100, n_val=20, n_epochs=2, n_workers=0, batch_size=8, prefetch_factor=2):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Declare encoders
    Encoder_SAD = EncoderSAD().to(device)
    Encoder_LTT = EncoderLTT().to(device)
    
    # Declare NPE
    NPE_builder = posterior_nn(
        model="maf_rqs",
        z_score_theta="none", #standardised on prior distribution moments because on-the-fly
        z_score_x="none",
        hidden_features=50,   #default
        num_transforms=5,     #default
        num_bins=10)          #default

    theta_init = torch.zeros((1, 4), device=device)
    embedding_init = torch.zeros((1, 64), device=device)

    NPE_emb_SAD = NPE_builder(theta_init, embedding_init)
    NPE_emb_LTT = NPE_builder(theta_init, embedding_init)
    
    # Declare joint models
    EcoEvo_emb_SAD = EcoEvoModel(Encoder_SAD, NPE_emb_SAD).to(device)
    EcoEvo_emb_LTT = EcoEvoModel(Encoder_LTT, NPE_emb_LTT).to(device)
    
    # Optimizers
    optim_emb_SAD = torch.optim.Adam(EcoEvo_emb_SAD.parameters(), lr=0.0005)
    optim_emb_LTT = torch.optim.Adam(EcoEvo_emb_LTT.parameters(), lr=0.0005)
    
    # Standardise theta manually because data are simulated on demand, tech from sbi see documentation
    # + voir moyenne et écart-type de la loi uniforme
    theta_mean = torch.tensor([(-7.0 - 3.0) / 2] + [(np.log10(SAMPLE_SIZE+1)+16.0)/2] * 3, dtype=torch.float32, device=device)
    theta_std = torch.tensor([4.0] + [16.0 - (np.log10(SAMPLE_SIZE+1))] * 3, dtype=torch.float32, device=device) / np.sqrt(12)
    

    # Initiate Datasets
    val_ids = range(1, 2*n_val, 2)
    val_dataset = SimDataset(val_ids)
    
    # DataLoader    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=n_workers,
        shuffle=False,
        drop_last=False,
        collate_fn=list,
        prefetch_factor=prefetch_factor if n_workers > 0 else None)
    
     
    # Initialise best val
    best_val_SAD = float("inf")
    best_val_LTT = float("inf")
    
    wait_times = []
    train_times = []
    ram_measurements = []
    
    # Epoch     
    for epoch in range(n_epochs):      
        
        EcoEvo_emb_SAD.train()
        EcoEvo_emb_LTT.train()
        
        # Initiate Datasets
        train_ids = range(2*epoch*n_train, 2*(epoch+1)*n_train, 2)
        train_dataset = SimDataset(train_ids)
            
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            num_workers=n_workers,
            shuffle=True,
            drop_last=True,
            collate_fn=list,
            prefetch_factor=prefetch_factor if n_workers > 0 else None)
        
        # Performance timers
        wait_start = time.perf_counter()
        
        # Batch
        for batch in train_loader:
            
            wait_time = time.perf_counter() - wait_start # time to fill a batch
            wait_times.append(wait_time)
            ram_measurements.append(total_ram())
            
            batch = [simulation for simulation in batch if simulation is not None]
    
            if not batch:
                continue
    
            
            theta, G, T, SAD, LTT = zip(*batch)
                # transforms the list batch = 
                # [(theta_1, G_1, T_1, SAD_1, LTT_1),
                #  (theta_2, G_2, T_2, SAD_2, LTT_2),
                #  ...]
                #
                # into tuples grouped by information level
                # theta = (theta_1, theta_2, ...)
                # SAD = (SAD_1, SAD_2, ...)
                # T = (T_1, T_2, ...)
            
            # Data curation
            SAD = torch.as_tensor(
                np.stack(SAD),
                dtype=torch.float32,
                device=device)
                # each SAD_i NumPy array of shape (SAMPLE_SIZE,)
                # np.stack(SAD) combines them into a single array of shape
                # (batch_size, SAMPLE_SIZE) with one simulation per row
                # Then convertion to tensor
            SAD = torch.log1p(SAD)
            
            theta = torch.as_tensor(
                np.stack(theta),
                dtype=torch.float32,
                device=device)
            
            theta_scaled = (theta-theta_mean)/theta_std
    
            LTT_padded = np.zeros((len(LTT), SAMPLE_SIZE), dtype=np.float32)
            for line, ltt in enumerate(LTT):
                LTT_padded[line, :len(ltt)] = np.log1p(ltt) # we fill each full-0 line in LTT_padded
                                                            # with a log1p(ltt) going from first date
                                                            # to len(ltt) and keeping full-0 padding
                                                            
            LTT = torch.as_tensor(
                LTT_padded,
                dtype=torch.float32,
                device=device).unsqueeze(1)
            
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            train_start = time.perf_counter()
            
            
            # Pass batch_SAD to models
            optim_emb_SAD.zero_grad() # reset gradients
            loss_emb_SAD = EcoEvo_emb_SAD.loss(theta_scaled, SAD) # pass SAD -> EncoderSAD -> NPE loss by comparing embedded parameters to real standardised parameters
            loss_emb_SAD.backward() # compute gradient
            optim_emb_SAD.step() # update weights of both networks
            
            # Pass batch_SAD to models
            optim_emb_LTT.zero_grad() # reset gradients
            loss_emb_LTT = EcoEvo_emb_LTT.loss(theta_scaled, LTT) # pass LTT -> EncoderLTT -> embedding (LTT) -> NPE posterior conditionned on embedding
            loss_emb_LTT.backward() # compute gradient
            optim_emb_LTT.step() # update weights of both networks

            if device.type == "cuda":
                torch.cuda.synchronize(device)
            train_time = time.perf_counter() - train_start
            train_times.append(train_time)

            print(f"Epoch {epoch+1}/{n_epochs} | "
                f"Train loss SAD: {loss_emb_SAD.item():.4f} | "
                f"Train loss LTT: {loss_emb_LTT.item():.4f}")
        
        wait_start = time.perf_counter()
        
        # Val    
        
        EcoEvo_emb_SAD.eval()
        EcoEvo_emb_LTT.eval()
        
        val_loss_SAD = 0.0
        val_loss_LTT = 0.0
        n_val_processed = 0
        
        with torch.no_grad():
            for batch in val_loader:
                
                theta, G, T, SAD, LTT = zip(*batch)
                SAD = torch.as_tensor(
                    np.stack(SAD),
                    dtype=torch.float32,
                    device=device)
                SAD = torch.log1p(SAD)
                
                theta = torch.as_tensor(
                    np.stack(theta),
                    dtype=torch.float32,
                    device=device)
                
                theta_scaled = (theta-theta_mean)/theta_std
    
                LTT_padded = np.zeros((len(LTT), SAMPLE_SIZE), dtype=np.float32)
                
                for line, ltt in enumerate(LTT):
                    LTT_padded[line, :len(ltt)] = np.log1p(ltt)
                
                LTT = torch.as_tensor(
                    LTT_padded,
                    dtype=torch.float32,
                    device=device).unsqueeze(1)
                
                loss_SAD = EcoEvo_emb_SAD.loss(theta_scaled, SAD)
                loss_LTT = EcoEvo_emb_LTT.loss(theta_scaled, LTT)
                
                n_batch = len(batch)
                val_loss_SAD += loss_SAD.item() * n_batch
                val_loss_LTT += loss_LTT.item() * n_batch
                
                n_val_processed += n_batch
            
            if n_val_processed == 0:
                raise RuntimeError("No validation simulation processed")
            
            val_loss_SAD /= n_val_processed
            val_loss_LTT /= n_val_processed
            
            print(f"Validation loss SAD: {val_loss_SAD:.4f}")
            print(f"Validation loss LTT: {val_loss_LTT:.4f}")
        
        if val_loss_SAD < best_val_SAD:
            best_val_SAD = val_loss_SAD
            torch.save(EcoEvo_emb_SAD.state_dict(), "best_SAD.pt")

        if val_loss_LTT < best_val_LTT:
            best_val_LTT = val_loss_LTT
            torch.save(EcoEvo_emb_LTT.state_dict(), "best_LTT.pt")
        
    return wait_times, train_times, max(ram_measurements, default=0.0)