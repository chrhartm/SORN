from __future__ import division
import numpy as np
import utils
utils.backup(__file__)

# see this file for parameter descriptions
from common.defaults import *

c.N_e = 200
c.N_i = np.floor(0.2*c.N_e)
c.N_u = 0
c.N = c.N_e + c.N_i

c.double_synapses = True
c.W_ee = utils.Bunch(use_sparse=True,
                     lamb=0.1*c.N_e,
                     avoid_self_connections=True,
                     eta_stdp = 0.004,
                     sp_prob = 0.1,
                     bias = 1.0, # no bias
                     p_failure = 0.2,
                     eta_ss = 0.1, # slow normalization
                     upper_bound = 1.0,
                     sp_initial=0.001)
                     
#~ W_ee_fail_f = lambda x: np.exp(-6*(x+0.2)) # 20% ca
W_ee_fail_f = lambda x: np.exp(-6*(x+0.1)) # can't be saved in bunch

c.W_ei = utils.Bunch(use_sparse=False,
                     lamb=0.2*c.N_e,
                     avoid_self_connections=False,
                     eta_istdp = 0.001,
                     h_ip=0.1)

c.W_ie = utils.Bunch(use_sparse=False,
                     lamb=np.inf,
                     avoid_self_connections=False)

c.steps_plastic = 1000000
c.steps_noplastic_train = 0
c.steps_noplastic_test = 0
c.N_steps = c.steps_plastic + c.steps_noplastic_train \
                            + c.steps_noplastic_test
c.display = True
c.N_iterations = 5
c.eta_ip = 0.01
c.h_ip = 0.1
c.T_e_max = 1.0
c.T_e_min = 0.0
c.T_i_max = 0.5
c.T_i_min = 0.0
c.synaptic_scaling = True
c.inhibitory_scaling = False


c.noise_sig = np.sqrt(0.05)
c.fast_inhibit = False
c.k_winner_take_all = False
c.ordered_thresholds = False

c.experiment.module = 'chartmann.alignment.experiment_alignment'
c.experiment.name = 'Experiment_alignment'

#######################################
c.stats.file_suffix = 'slowSN'
#######################################
c.stats.rand_networks = 0

from common.sources import NoSource
source = NoSource()

# Cluster  
c.cluster.vary_param = 'W_ee.p_failure'
c.cluster.params = [0.1,0.2,0.3]
if c.imported_mpi:
    c.cluster.NUMBER_OF_SIMS  = len(c.cluster.params)
    c.cluster.NUMBER_OF_CORES = MPI.COMM_WORLD.size
    c.cluster.NUMBER_LOCAL = c.cluster.NUMBER_OF_SIMS\
                             // c.cluster.NUMBER_OF_CORES
