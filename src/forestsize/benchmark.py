import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import psutil


# Valeur utilisée quand tu lances toi-même la campagne depuis le terminal.
trial_config = None

if __name__ == "__main__" and sys.argv[1:2] == ["--trial"]:
    # Ici, on est dans le processus chargé d'un seul essai.
    if len(sys.argv) != 9:
        raise ValueError("Sept valeurs sont attendues après --trial")

    trial_config = {
        "cpus": int(sys.argv[2]),
        "workers": int(sys.argv[3]),
        "batch_size": int(sys.argv[4]),
        "prefetch": None if sys.argv[5] == "None" else int(sys.argv[5]),
        "ram_reserved": float(sys.argv[6]),
        "n_train": int(sys.argv[7]),
        "n_val": int(sys.argv[8]),
    }

    process = psutil.Process()
    allowed_cpus = sorted(process.cpu_affinity())
    requested_cpus = trial_config["cpus"]

    # Sur cluster, partir des CPU déjà accessibles dans l'allocation.
    if not 1 <= requested_cpus <= len(allowed_cpus):
        raise ValueError(
            f"{requested_cpus} CPU demandés, {len(allowed_cpus)} accessibles"
        )

    # Appliquer la limite avant l'import de PyTorch et la création des workers.
    process.cpu_affinity(allowed_cpus[:requested_cpus])


import torch
from forestsize.inference import ModelTraining

def monitor_ram(stop, interval=0.5):
    parent = psutil.Process() # running process
    metric = "pss" if sys.platform.startswith("linux") else "uss"
    peak_bytes = None
    
    while True: # until main process ends
        processes = [parent, *parent.children(recursive=True)] # create a list of workers ids to sum their memory usage (linked to parent = main process)
        total_bytes = 0
        complete_sample = True
        
        for process in processes:
            try:
                info = process.memory_full_info() # get worker statistics
                worker_bytes = getattr(info, metric)
                total_bytes += worker_bytes
            except (psutil.NoSuchProcess):
                continue
            except (psutil.AccessDenied, AttributeError):
                complete_sample = False
                break
            
        if complete_sample:
            if peak_bytes is None or total_bytes > peak_bytes:
                peak_bytes = total_bytes
                
        if stop.wait(interval):
            break
        
    peak_mib = None if peak_bytes is None else peak_bytes / 2**20
    
    return metric.upper(), peak_mib

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
                        
                    if trial_config is None:
                        subprocess.run(
                            [
                                sys.executable, __file__, "--trial",
                                str(cpus), str(workers), str(batch_size),
                                str(prefetch), str(ram_reserved),
                                str(n_train), str(n_val),
                            ],
                            check=True,
                        )
                        continue
                    
                    train_batches = (n_train + batch_size - 1) // batch_size
                
                    print(f"CPU logiques autorisés : {cpus} | "
                          f"workers : {workers} | "
                          f"batch : {batch_size} | "
                          f"budget RAM envisagé : {ram_reserved} Gio | "
                          f"simulations train/val : {n_train}/{n_val} | "
                          f"mises à jour par NPE : {train_batches}")
                    
                    torch.manual_seed(123456)
                    if not torch.cuda.is_available():
                        raise RuntimeError("No GPU available")
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                    
                    stop = threading.Event()
                    
                    with ThreadPoolExecutor(max_workers=1) as observer:
                        
                        ram_monitor = observer.submit(monitor_ram, stop)
                    
                        training_call_start = time.perf_counter()
                        
                        try:
                           first_batch_wait, later_batches_wait =  ModelTraining(
                                                                        n_train=n_train,
                                                                        n_val=n_val,
                                                                        n_workers=workers,
                                                                        batch_size=batch_size,
                                                                        prefetch_factor=prefetch,
                                                                        benchmark_run=True)
                           torch.cuda.synchronize()
                        
                           training_call_s = time.perf_counter() - training_call_start
                        
                        finally:
                            stop.set()
                        ram_metric, peak_ram_mib = ram_monitor.result()
                        
                    vram_allocated_mib = torch.cuda.max_memory_allocated() / 2**20
                    vram_reserved_mib = torch.cuda.max_memory_reserved() / 2**20

                    if peak_ram_mib is None:
                        ram_text = "indisponible"
                    else:
                        ram_text = f"{peak_ram_mib:.1f} Mio ({ram_metric})"

                    print(
                        f"prefetch={prefetch} | "
                        f"entraînement + validation={training_call_s:.2f} s | "
                        f"débit de bout en bout={n_train / training_call_s:.3f} simulations train/s | "
                        f"pic RAM observé={ram_text} | "
                        f"pic VRAM allouée={vram_allocated_mib:.1f} Mio | "
                        f"pic VRAM réservée={vram_reserved_mib:.1f} Mio | "
                        f"attente premier batch={first_batch_wait:.2f} s | "
                        f"attente batches suivants (somme)={later_batches_wait:.2f} s"
                    )          
                



if __name__ == "__main__":
    if trial_config is None:
        # Tu as lancé benchmark.py toi-même : parcourir la grille demandée.
        benchmark(
            cpu_counts=[2, 4],
            worker_counts=[1, 2],
            batch_sizes=[2, 4],
            ram_reserved=32,
            n_train=128,
            n_val=32,
            prefetch_factors=[1, 2],
        )
    else:
        # Ce processus a été lancé pour UNE combinaison précise.
        benchmark(
            cpu_counts=[trial_config["cpus"]],
            worker_counts=[trial_config["workers"]],
            batch_sizes=[trial_config["batch_size"]],
            ram_reserved=trial_config["ram_reserved"],
            n_train=trial_config["n_train"],
            n_val=trial_config["n_val"],
            prefetch_factors=[trial_config["prefetch"]],
        )