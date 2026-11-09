import copy
import numpy as np
import ecophylo
from ecophylo.phylogen import toPhylo
from forestsize.parameters import sample_parameters, log_writer, MASTER_SEED, SAMPLE_SIZE


def genealogy(sim_id):
    
    theta, sample_size, changetimes, com_size, seeds = sample_parameters(sim_id)
    
    if theta is None or com_size is None:
        # No need to call log_writer as it was already called in parameters.py
        return None, None, None
    
    try:
        G = ecophylo.simulate(
            samples = sample_size,
            deme_sizes = com_size,
            mu = 10**theta[0],
            changetimes = changetimes,
            spmodel = "genealogy",
            tau = 0,
            seed = seeds["G_seed"],
            verbose = False)
        
    except Exception as error:
        
        last_error = error
        
        while seeds["retries"]:
            G_seed = seeds["retries"].pop(0)
            try:
                G = ecophylo.simulate(
                    samples = sample_size,
                    deme_sizes = com_size,
                    mu = 10**theta[0],
                    changetimes = changetimes,
                    spmodel = "genealogy",
                    tau = 0,
                    seed = int(G_seed),
                    verbose = False)
                
                seeds["G_seed"] = int(G_seed)
                break
            
            except Exception as retry_error:
                last_error = retry_error
        else:
            log_writer(
                error_type= f'Genealogy error : {last_error}',
                master_seed = MASTER_SEED,
                sim_id = sim_id,
                theta = theta,
                seeds = seeds)
            raise RuntimeError("All genealogy retries failed") from last_error
    
    return theta, G, seeds


def speciation(sim_id):
    
    theta, G, seeds = genealogy(sim_id)
    
    if G is None:
        # No need to call log_writer as it was already called in genealogy()
        return theta, None, None, seeds
    
    try:
        T = toPhylo(
            tree = copy.deepcopy(G),
            mu = 10**theta[0],
            tau = 0,
            spmodel = "loose",
            seed = seeds["T_seed"])
        
    except Exception as error:
        
        last_error = error
        
        while seeds["retries"]:
            T_seed = seeds["retries"].pop(0)
            try:
                T = toPhylo(
                    tree = copy.deepcopy(G),
                    mu = 10**theta[0],
                    tau = 0,
                    spmodel = "loose",
                    seed = int(T_seed))
                
                seeds["T_seed"] = int(T_seed)
                break
            
            except Exception as retry_error:
                last_error = retry_error
        else:
            log_writer(
                error_type = f'Speciation error : {last_error}',
                master_seed = MASTER_SEED,
                sim_id = sim_id,
                theta = theta,
                seeds = seeds)
            raise RuntimeError("All phylogenies retries failed") from last_error
    
    return theta, G, T, seeds


def SumStat(sim_id):

    theta, G, T, seeds = speciation(sim_id)

    if T is None:
        raise RuntimeError(f"Simulation {sim_id}: no speciation tree")

    while True:

        try:
            if any(a < 1 or a > SAMPLE_SIZE for a in ecophylo.getAbund(T)):
                raise ValueError("Invalid Abundance")

            SAD = ecophylo.getSAD(T, method="exact")
            LTT = np.array(ecophylo.getLTT(T).get("time"), dtype=np.float64)

            return theta, G, T, SAD, LTT, seeds

        except Exception as error:
            last_error = error

        # Rebuild T using the remaining retry seeds
        while seeds["retries"]:

            T_seed = int(seeds["retries"].pop(0))

            try:
                T = toPhylo(
                    tree=copy.deepcopy(G),
                    mu=10**theta[0],
                    tau=0,
                    spmodel="loose",
                    seed=T_seed)

                seeds["T_seed"] = T_seed
                break

            except Exception as retry_error:
                last_error = retry_error

        else:
            log_writer(
                error_type=f"SumStat error after retries: {last_error}",
                master_seed=MASTER_SEED,
                sim_id=sim_id,
                theta=theta,
                seeds=seeds)

            raise RuntimeError("All SumStat retries failed") from last_error