import copy
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
            mu = theta[0],
            changetimes = changetimes,
            spmodel = "genealogy",
            tau = 0,
            seed = seeds["G_seed"],
            verbose = False
        )
    except Exception as error:
        log_writer(
            error_type= f'Genealogy error : {error}',
            master_seed = MASTER_SEED,
            sim_id = sim_id,
            theta = theta,
            seeds = seeds)
        
        return theta, None, seeds
    
    return theta, G, seeds


def speciation(sim_id):
    
    theta, G, seeds = genealogy(sim_id)
    
    if G is None:
        # No need to call log_writer as it was already called in oracle()
        return None
    
    try:
        T = toPhylo(
            tree = copy.deepcopy(G),
            mu = theta[0],
            tau = 0,
            spmodel = "loose",
            seed = seeds["T_seed"])
        
    except Exception as error:
        log_writer(
            error_type = f'Speciation error : {error}',
            master_seed = MASTER_SEED,
            sim_id = sim_id,
            theta = theta,
            seeds = seeds)
        
        return theta, None, seeds
    
    return theta, G, T, seeds


def SumStat(sim_id):
    
    theta, G, T, seeds = speciation(sim_id)
    
    if T is None:
        return None, None
    
    if any(a < 1 or a > SAMPLE_SIZE for a in ecophylo.getAbund(T)):
        log_writer(
            error_type = "Invalid Abundance",
            master_seed = MASTER_SEED,
            sim_id = sim_id,
            theta = theta,
            seeds = seeds)
        
        return None, None
        
    SAD = ecophylo.getSAD(T, method="exact")
    LTT = ecophylo.getLTT(T)
    
    return theta, G, T, SAD, LTT, seeds