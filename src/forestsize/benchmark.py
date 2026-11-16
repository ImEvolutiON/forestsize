import time
import torch
from forestsize.inference import ModelTraining

def benchmark(cpu_counts, worker_counts, batch_sizes, ram_reserved, prefetch_factors, *, n_train, n_val):
    for cpus in cpu_counts:
        for workers in worker_counts:
            for batch_size in batch_sizes:
                factors = prefetch_factors if workers > 0 else [None]
                for prefetch in factors:
                    
                    if type(batch_size) is not int or batch_size < 1:
                        raise ValueError("Error batch_size is less than 1")            
                    if type(n_train) is not int or n_train < 1:
                        raise ValueError("Error n_train is less than 1")
                    if type(n_val) is not int or n_val < 1:
                        raise ValueError("Error n_val is less than 1") 
                    if n_train % batch_size != 0:
                        raise ValueError("Error n_train isn't a positive multiplier of batch_size, you may compare batches of different sizes")
                        
                    train_batches = n_train // batch_size
                    
                
                    print(f"CPU utilisés : {cpus} | "
                          f"workers : {workers} | "
                          f"batch : {batch_size} | "
                          f"RAM réservée : {ram_reserved} Go | "
                          f"simulations train/val : {n_train}/{n_val} | "
                          f"mises à jour par NPE : {train_batches}")
                    
                    torch.manual_seed(123456)
                    if not torch.cuda.is_available():
                        raise RuntimeError("No GPU available")
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                    
                    start = time.perf_counter()
                    
                    metrics = ModelTraining(
                        n_train=n_train,
                        n_val=n_val,
                        n_epochs=1,
                        n_workers=workers,
                        batch_size=batch_size,
                        prefetch_factor=prefetch,
                        benchmark_run=True)
                    
                    elapsed = time.perf_counter() - start
                    
                    torch.cuda.synchronize()
                    elapsed = time.perf_counter() - start
                    vram_allocated_mib = torch.cuda.max_memory_allocated() / 2**20
                    vram_reserved_mib = torch.cuda.max_memory_reserved() / 2**20
                    
                    train_ids = metrics["train_ids"]
                    train_steps_s = metrics["train_steps_s"]

                    print(
                        f"prefetch={prefetch} | "
                        f"débit global={train_ids / elapsed:.3f} tirages/s | "
                        f"débit pendant les mises à jour={train_ids / train_steps_s:.3f} tirages/s | "
                        f"attente premier batch={metrics['first_batch_wait_s']:.2f} s | "
                        f"attente batches suivants={metrics['later_batch_wait_s']:.2f} s"
                        f" | pic VRAM allouée={vram_allocated_mib:.1f} Mio"
                        f" | pic VRAM réservée={vram_reserved_mib:.1f} Mio"
                    )
                
                
                



if __name__ == "__main__":
    benchmark(
        cpu_counts=[2],
        worker_counts=[1],
        batch_sizes=[2, 4],
        ram_reserved=32,
        n_train=16,
        n_val=2,
        prefetch_factors=[1],
    )