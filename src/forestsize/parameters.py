import numpy as np
import csv
import sys


MASTER_SEED = 123456
SAMPLE_SIZE = 50000
CHANGETIMES = [[0, 98, 677]]

def sample_parameters(sim_id):
    
    rng = np.random.default_rng(np.random.SeedSequence([MASTER_SEED, sim_id]))
    
    mu = 10 ** rng.uniform(-7.0, -3.0)
    
    N0 = 10 ** rng.uniform(3.0, 16.0)
    N1 = 10 ** rng.uniform(3.0, 16.0)
    N2 = 10 ** rng.uniform(3.0, 16.0)
    
    theta = np.array([mu, N0, N1, N2], dtype = np.float64)
    
    com_size = [[N0, N1, N2]]
    
    if min(com_size[0]) < SAMPLE_SIZE:
        log_writer(
            error_type="Invalid Prior : Ne(t) contains a value smaller than the sampling size",
            master_seed = MASTER_SEED,
            sim_id = sim_id,
            theta = theta)
        
        return None, SAMPLE_SIZE, CHANGETIMES, com_size, None
    
    G_seed = int(rng.integers(1, 2**32))
    T_seed = int(rng.integers(1, 2**32))
    ind_sampling_seed = int(rng.integers(1, 2**32))
    seeds = {"G_seed": G_seed,
             "T_seed": T_seed,
             "ind_sampling_seed": ind_sampling_seed}
    
    return theta, SAMPLE_SIZE, CHANGETIMES, com_size, seeds


def log_writer(error_type, master_seed, sim_id, theta, seeds=None):
    mu, N0, N1, N2 = theta
    
    if seeds is None:
        seeds = {}
    
    writer = csv.writer(sys.stderr)
    writer.writerow([error_type,
                     master_seed,
                     sim_id,
                     
                     SAMPLE_SIZE,
                     CHANGETIMES,
                     
                     mu,
                     N0,
                     N1,
                     N2,
                     
                     seeds.get("G_seed"),
                     seeds.get("T_seed"),
                     seeds.get("ind_sampling_seed")])