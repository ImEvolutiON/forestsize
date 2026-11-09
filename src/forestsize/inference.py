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
from forestsize.encoders import EncoderSAD, EncoderLTT, ConcatFeatures



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

    # Save benchmark results after each configuration
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
    Encoder_embSAD = EncoderSAD().to(device) # embSAD
    
    Encoder_embSAD_joint_embLTT = EncoderSAD().to(device) # embSAD+embLTT
    Encoder_embLTT_joint_embSAD = EncoderLTT().to(device) # embSAD+embLTT
    Encoder_embSAD_joint_embLTT_concat = ConcatFeatures(Encoder_embSAD_joint_embLTT, Encoder_embLTT_joint_embSAD) # joint embSAD+embLTT
    
    Encoder_rawSAD = torch.nn.Identity() # rawSAD alone
    
    Encoder_rawSAD_joint_embLTT = torch.nn.Identity() # rawSAD+embLTT
    Encoder_embLTT_joint_rawSAD = EncoderLTT().to(device) # rawSAD+embLTT
    Encoder_rawSAD_joint_embLTT_concat = ConcatFeatures(Encoder_rawSAD_joint_embLTT, Encoder_embLTT_joint_rawSAD) # joint rawSAD+embLTT
    
    # Declare NPE
    NPE_builder = posterior_nn(
        model="maf_rqs",
        z_score_theta="none", #standardised on prior distribution moments because on-the-fly
        z_score_x="none",
        hidden_features=50,   #default
        num_transforms=5,     #default
        num_bins=10)          #default

    theta_init = torch.zeros((1, 4), device=device)
    embSAD_init = torch.zeros((1, 64), device=device)
    embSAD_joint_embLTT_init = torch.zeros((1, 128), device=device)
    rawSAD_init = torch.zeros((1, SAMPLE_SIZE), device=device)
    rawSAD_joint_embLTT_init = torch.zeros((1, SAMPLE_SIZE + 64), device=device)

    NPE_embSAD = NPE_builder(theta_init, embSAD_init).to(device)
    NPE_embSAD_joint_embLTT = NPE_builder(theta_init, embSAD_joint_embLTT_init).to(device)
    NPE_rawSAD = NPE_builder(theta_init, rawSAD_init).to(device)
    NPE_rawSAD_joint_embLTT = NPE_builder(theta_init, rawSAD_joint_embLTT_init).to(device)
    
    # Declare joint models
    EcoEvo_embSAD = EcoEvoModel(Encoder_embSAD, NPE_embSAD).to(device)
    EcoEvo_embSAD_joint_embLTT = EcoEvoModel(Encoder_embSAD_joint_embLTT_concat, NPE_embSAD_joint_embLTT)
    EcoEvo_rawSAD = EcoEvoModel(Encoder_rawSAD, NPE_rawSAD)
    EcoEvo_rawSAD_joint_embLTT = EcoEvoModel(Encoder_rawSAD_joint_embLTT_concat, NPE_rawSAD_joint_embLTT)
    
    # Optimizers
    optim_embSAD = torch.optim.Adam(EcoEvo_embSAD.parameters(), lr=0.0005)
    optim_embSAD_joint_embLTT = torch.optim.Adam(EcoEvo_embSAD_joint_embLTT.parameters(), lr=0.0005)
    optim_rawSAD = torch.optim.Adam(EcoEvo_rawSAD.parameters(), lr=0.0005)
    optim_rawSAD_joint_embLTT = torch.optim.Adam(EcoEvo_rawSAD_joint_embLTT.parameters(), lr=0.0005)
    
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
    best_val_embSAD = float("inf")
    best_val_embSAD_joint_embLTT = float("inf")
    best_val_rawSAD = float("inf")
    best_val_rawSAD_joint_embLTT = float("inf")
    
    wait_times = []
    train_times = []
    ram_measurements = []
    
    # Epoch     
    for epoch in range(n_epochs):      
        
        EcoEvo_embSAD.train()
        EcoEvo_embSAD_joint_embLTT.train()
        EcoEvo_rawSAD.train()
        EcoEvo_rawSAD_joint_embLTT.train()
        
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
    
            # Data curations
            theta, train_G, train_T, train_SAD, train_LTT = zip(*batch)
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
            train_SAD = torch.as_tensor(
                np.stack(train_SAD),
                dtype=torch.float32,
                device=device)
                # each SAD_i NumPy array of shape (SAMPLE_SIZE,)
                # np.stack(SAD) combines them into a single array of shape
                # (batch_size, SAMPLE_SIZE) with one simulation per row
                # Then convertion to tensor
            train_SAD = torch.log1p(train_SAD)
            
            theta = torch.as_tensor(
                np.stack(theta),
                dtype=torch.float32,
                device=device)
            
            theta_scaled = (theta-theta_mean)/theta_std
    
            train_LTT_padded = np.zeros((len(train_LTT), SAMPLE_SIZE), dtype=np.float32)
            for line, ltt in enumerate(train_LTT):
                train_LTT_padded[line, :len(ltt)] = np.log1p(ltt) # we fill each full-0 line in LTT_padded
                                                            # with a log1p(ltt) going from first date
                                                            # to len(ltt) and keeping full-0 padding
                                                            
            train_LTT = torch.as_tensor(
                train_LTT_padded,
                dtype=torch.float32,
                device=device).unsqueeze(1)
            
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            train_start = time.perf_counter()
            
            
            # Pass batch_embSAD to models
            optim_embSAD.zero_grad() # reset gradients
            loss_embSAD = EcoEvo_embSAD.loss(theta_scaled, train_SAD) # pass SAD -> EncoderSAD -> NPE loss by comparing embedded parameters to real standardised parameters
            loss_embSAD.backward() # compute gradient
            optim_embSAD.step() # update weights of both networks
            
            # Pass batch_embSAD_embLTT_joint to models
            optim_embSAD_joint_embLTT.zero_grad() # reset gradients
            loss_embSAD_joint_embLTT = EcoEvo_embSAD_joint_embLTT.loss(theta_scaled, (train_SAD, train_LTT)) # pass LTT -> EncoderLTT -> embedding (LTT) -> NPE posterior conditionned on embedding
            loss_embSAD_joint_embLTT.backward() # compute gradient
            optim_embSAD_joint_embLTT.step() # update weights of both networks
            
            # Pass batch_rawSAD to models
            optim_rawSAD.zero_grad() # reset gradients
            loss_rawSAD = EcoEvo_rawSAD.loss(theta_scaled, train_SAD) # pass SAD -> EncoderSAD -> NPE loss by comparing embedded parameters to real standardised parameters
            loss_rawSAD.backward() # compute gradient
            optim_rawSAD.step() # update weights of both networks
            
            # Pass batch_rawSAD_embLTT_joint to models
            optim_rawSAD_joint_embLTT.zero_grad() # reset gradients
            loss_rawSAD_joint_embLTT = EcoEvo_rawSAD_joint_embLTT.loss(theta_scaled, (train_SAD, train_LTT)) # pass LTT -> EncoderLTT -> embedding (LTT) -> NPE posterior conditionned on embedding
            loss_rawSAD_joint_embLTT.backward() # compute gradient
            optim_rawSAD_joint_embLTT.step() # update weights of both networks

            if device.type == "cuda":
                torch.cuda.synchronize(device)
            train_time = time.perf_counter() - train_start
            train_times.append(train_time)

            print(f"Epoch {epoch+1}/{n_epochs} | "
                f"Train loss embedding SAD: {loss_embSAD.item():.4f} | "
                f"Train loss embedding SAD + embedding LTT: {loss_embSAD_joint_embLTT.item():.4f} | "
                f"Train loss raw SAD: {loss_rawSAD.item():.4f} | "
                f"Train loss raw SAD + embedding LTT: {loss_rawSAD_joint_embLTT.item():.4f}")
        
            wait_start = time.perf_counter()
        
        # Validation
        # After all batches are done, do one val check for the epoch
        
        # Disable training behaviour of the models
        EcoEvo_embSAD.eval()
        EcoEvo_embSAD_joint_embLTT.eval()
        EcoEvo_rawSAD.eval()
        EcoEvo_rawSAD_joint_embLTT.eval()
        
        # Initialise cumulative validation losses
        val_loss_embSAD = 0.0
        val_loss_embSAD_joint_embLTT = 0.0
        val_loss_rawSAD = 0.0
        val_loss_rawSAD_joint_embLTT = 0.0
        n_val_processed = 0
        
        # Disable gradient computation during validation
        with torch.no_grad():
            for batch in val_loader:
                
                # Data curations
                theta, val_G, val_T, val_SAD, val_LTT = zip(*batch)
                val_SAD = torch.as_tensor(
                    np.stack(val_SAD),
                    dtype=torch.float32,
                    device=device)
                val_SAD = torch.log1p(val_SAD)
                
                theta = torch.as_tensor(
                    np.stack(theta),
                    dtype=torch.float32,
                    device=device)
                
                theta_scaled = (theta-theta_mean)/theta_std
    
                val_LTT_padded = np.zeros((len(val_LTT), SAMPLE_SIZE), dtype=np.float32)
                
                for line, ltt in enumerate(val_LTT):
                    val_LTT_padded[line, :len(ltt)] = np.log1p(ltt)
                
                val_LTT = torch.as_tensor(
                    val_LTT_padded,
                    dtype=torch.float32,
                    device=device).unsqueeze(1)
                
                # Compute the mean loss of each model on the current validation batch
                loss_embSAD = EcoEvo_embSAD.loss(theta_scaled, val_SAD)
                loss_embSAD_joint_embLTT = EcoEvo_embSAD_joint_embLTT.loss(theta_scaled, (val_SAD, val_LTT))
                loss_rawSAD = EcoEvo_rawSAD.loss(theta_scaled, val_SAD)
                loss_rawSAD_joint_embLTT = EcoEvo_rawSAD_joint_embLTT.loss(theta_scaled, (val_SAD, val_LTT))
                
                # Accumulate losses weighted by the number of simulations in the batch
                n_batch = len(batch)
                val_loss_embSAD += loss_embSAD.item() * n_batch
                val_loss_embSAD_joint_embLTT += loss_embSAD_joint_embLTT.item() * n_batch
                val_loss_rawSAD += loss_rawSAD.item() * n_batch
                val_loss_rawSAD_joint_embLTT += loss_rawSAD_joint_embLTT.item() * n_batch
                
                n_val_processed += n_batch
            
            if n_val_processed == 0:
                raise RuntimeError("No validation simulation processed")
            
            # Mean losses over all validations
            val_loss_embSAD /= n_val_processed
            val_loss_embSAD_joint_embLTT /= n_val_processed
            val_loss_rawSAD /= n_val_processed
            val_loss_rawSAD_joint_embLTT /= n_val_processed
            
            print(f"Validation loss embedding SAD: {val_loss_embSAD:.4f}")
            print(f"Validation loss embedding SAD + embedding LTT: {val_loss_embSAD_joint_embLTT:.4f}")
            print(f"Validation loss raw SAD: {val_loss_rawSAD:.4f}")
            print(f"Validation loss raw SAD + embedding LTT: {val_loss_rawSAD_joint_embLTT:.4f}")
        
        # Save model weights when validation loss improves
        if val_loss_embSAD < best_val_embSAD:
            best_val_embSAD = val_loss_embSAD
            torch.save(EcoEvo_embSAD.state_dict(), "best_embSAD.pt")            
            
        if val_loss_embSAD_joint_embLTT < best_val_embSAD_joint_embLTT:
            best_val_embSAD_joint_embLTT = val_loss_embSAD_joint_embLTT
            torch.save(EcoEvo_embSAD_joint_embLTT.state_dict(), "best_embSAD_joint_embLTT.pt")
        
        if val_loss_rawSAD < best_val_rawSAD:
            best_val_rawSAD = val_loss_rawSAD
            torch.save(EcoEvo_rawSAD.state_dict(), "best_rawSAD.pt")
            
        if val_loss_rawSAD_joint_embLTT < best_val_rawSAD_joint_embLTT:
            best_val_rawSAD_joint_embLTT = val_loss_rawSAD_joint_embLTT
            torch.save(EcoEvo_rawSAD_joint_embLTT.state_dict(), "best_rawSAD_joint_embLTT.pt")
            
    return wait_times, train_times, max(ram_measurements, default=0.0)