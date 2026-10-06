import sys
import ecophylo
import numpy as np
import csv

MASTER_SEED = 123456789
MAX_ABUND = 50000
SAD_BASE = 1.1

x = np.arange(1, MAX_ABUND + 1)
y = (np.log(x) / np.log(SAD_BASE)).round().astype(int)
_, x_to_bin = np.unique(y, return_inverse=True)
n_sad_bin = x_to_bin.max()+1


def log_writer(error_type, sim_id, mu_seed, No_seed, k1_seed, k2_seed, tree_seed, mu, No, k1, k2):
    writer = csv.writer(sys.stderr)
    writer.writerow([error_type,
                     sim_id, 
                     mu_seed, 
                     No_seed, 
                     k1_seed, 
                     k2_seed, 
                     tree_seed, 
                     mu, 
                     No, 
                     k1, 
                     k2])
    

def simulation(sim_id):
    
    rng = np.random.default_rng(np.random.SeedSequence([MASTER_SEED, sim_id]))
    
    while True:
        
        mu_seed = int(rng.integers(1, 2**32))
        mu = ecophylo.sample(
            lower = 1e-7,
            upper = 1e-3,
            distr = "log_unif",
            seed = mu_seed
        )
        
        No_seed = int(rng.integers(1, 2**32))
        No = ecophylo.sample(
            lower = 1e7,
            upper = 1e12,
            distr = "log_unif",
            seed = No_seed
        )
        
        k1_seed = int(rng.integers(1, 2**32))
        k1 = ecophylo.sample(
            lower = 0.01,
            upper = 100,
            distr = "log_unif",
            seed = k1_seed
            )
        
        k2_seed = int(rng.integers(1, 2**32))
        k2 = ecophylo.sample(
            lower = 0.01,
            upper = 100,
            distr = "log_unif",
            seed = k2_seed
            )
        
        deme_sizes = [[
            No,
            No * k1,
            No * k1 * k2
        ]]
        
        if min(deme_sizes[0]) < MAX_ABUND:
            log_writer(error_type = "Ne_lower_than_sampling",
                       sim_id = sim_id,
                       mu_seed = mu_seed,
                       No_seed = No_seed,
                       k1_seed = k1_seed,
                       k2_seed = k2_seed,
                       tree_seed = "None",
                       mu = mu,
                       No = No,
                       k1 = k1,
                       k2 = k2)
            continue
        
        changetimes = [[0, 98, 677]]
        
        tree_seed = int(rng.integers(1, 2**32))
        tree = ecophylo.simulate(
            samples = MAX_ABUND,
            deme_sizes = deme_sizes,
            mu = mu,
            changetimes = changetimes,
            spmodel = "loose",
            tau = 0,
            seed = tree_seed,
            verbose = False
        )
                
        abund = np.array(ecophylo.getAbund(tree))
        
        if np.any((abund < 1) | (abund > MAX_ABUND)):
            log_writer(error_type = "Invalid_abundance",
                       sim_id = sim_id,
                       mu_seed = mu_seed,
                       No_seed = No_seed,
                       k1_seed = k1_seed,
                       k2_seed = k2_seed,
                       tree_seed = tree_seed,
                       mu = mu,
                       No = No,
                       k1 = k1,
                       k2 = k2)
            continue
        
        bin_ids = x_to_bin[abund - 1]
        sad = np.bincount(
            bin_ids,
            minlength = n_sad_bin
            ).astype(np.float32)
        
        params = np.array([mu, No, k1, k2], dtype=np.float32)
    
        return params, tree, sad