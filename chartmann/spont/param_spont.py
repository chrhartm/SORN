from __future__ import division
import numpy as np
import utils
utils.backup(__file__)

# see this file for parameter descriptions
from common.defaults import *

c.N_e = 200
c.N_i = int(np.floor(0.2*c.N_e))
c.N = c.N_e + c.N_i
c.N_u_e = np.floor(0.05*c.N_e)
c.N_u_i = 0

c.double_synapses = False
c.W_ee = utils.Bunch(use_sparse=True,
                     lamb=0.1*c.N_e,
                     avoid_self_connections=True,
                     eta_stdp = 0.001,
                     sp_prob = 0.0,
                     sp_initial=0.000,
                     no_prune = True,
                     upper_bound = 1,   
                     weighted_stdp = False,
                     eta_ds = 0.1
                     )

c.W_ei = utils.Bunch(use_sparse=False,
                     lamb=np.inf,
                     avoid_self_connections=False,
                     eta_istdp = 0.0,
                     h_ip=0.1)

c.W_ie = utils.Bunch(use_sparse=False,
                     lamb=np.inf,
                     avoid_self_connections=False)

c.steps_plastic = 50000
c.steps_noplastic_train = 20000
c.steps_noplastic_test = 50000
c.N_steps = c.steps_plastic + c.steps_noplastic_train \
                            + c.steps_noplastic_test
c.display = True

c.N_iterations = 20

# 0.1 -> ~30% noisespikes, 0.05 -> ~15%, 0.01 -> ~2.5%, 0.005 -> ~1% 
c.noise_sig = 0

c.with_plasticity = True

c.input_gain = 0.5

c.eta_ip = 0.001
h_ip_mean = float(2*c.N_u_e)/float(c.N_e)
h_ip_range = 0.01
c.h_ip = np.random.rand(c.N_e)*h_ip_range*2 + h_ip_mean - h_ip_range
c.always_ip = True
c.synaptic_scaling = True

c.T_e_max = 0.5
c.T_e_min = 0.0
c.T_i_max = 0.35
c.T_i_min = 0.0
c.ordered_thresholds = True

c.fast_inhibit = True
c.k_winner_take_all = False
c.ff_inhibition = False
c.ff_inhibition_broad = 0.0

c.experiment.module = 'chartmann.spont.experiment_spont'
c.experiment.name = 'Experiment_spont'

#######################################
c.stats.file_suffix = 'ABCD_ds_noSTDP'
#######################################
c.stats.save_spikes = True
c.stats.quenching = 'train'
c.stats.quenching_window = 2
c.stats.match = False
c.stats.lstsq_mue = 1
c.stats.control_rates = False
c.stats.ISI_step = 4

# Following parameters for randsource
c.source.use_randsource = False
c.source.N_words = 10
c.source.N_letters = 10
c.source.word_length = np.array([2,6])
c.source.max_fold_prob = 3

c.source.prob = 0.75 # This is only here to be changed by cluster
c.source.avoid = False
c.source.control = False # For sequence_test

# For cluster: same randsource for entire simulation
#~ from common.sources import CountingSource
#~ source = CountingSource.init_simple(
        #~ c.source.N_words,
        #~ c.source.N_letters,c.source.word_length,
        #~ c.source.max_fold_prob,c.N_u_e,c.N_u_i,
        #~ c.source.avoid, seed=42)
#~ import random
#~ print source.words, np.random.randint(0,500), random.random()

#~ from common.sources import RandomLetterSource
#~ source = RandomLetterSource(c.source.N_letters,c.N_u_e,c.N_u_i,
                            #~ c.source.avoid)
from common.sources import CountingSource
source = CountingSource(['ABCD','EFGH'],
                         np.array([[2./3., 1./3.],
                                   [2./3., 1./3.]]),
             c.N_u_e,c.N_u_i,c.source.avoid)
                        
c.wait_min_plastic = 0
c.wait_var_plastic = 0
c.wait_min_train = 0
c.wait_var_train = 0
                        
# Cluster
c.cluster.vary_param = 'source.prob'#'with_plasticity'#
c.cluster.params = np.linspace(0.1,0.9,11)#[False,True]#
if c.imported_mpi:
    c.cluster.NUMBER_OF_SIMS  = len(c.cluster.params)
    c.cluster.NUMBER_OF_CORES = MPI.COMM_WORLD.size
    c.cluster.NUMBER_LOCAL = c.cluster.NUMBER_OF_SIMS\
                             // c.cluster.NUMBER_OF_CORES
