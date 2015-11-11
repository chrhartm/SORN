from __future__ import division
import numpy as np
import utils
utils.backup(__file__)

# see this file for parameter descriptions
from common.defaults import *

c.N_e = 200
c.N_i = np.floor(0.2*c.N_e)
c.N_u = 0 #noinput is the point of this experiment
c.N = c.N_e + c.N_i

c.W_ee = utils.Bunch(use_sparse=True,
                     lamb=0.1*c.N_e,
                     avoid_self_connections=True,
                     eta_stdp = 0.004,
                     sp_prob = 0.1,
                     sp_initial=0.001)

c.W_ei = utils.Bunch(use_sparse=False,
                     lamb=0.2*c.N_e,
                     avoid_self_connections=False,
                     eta_istdp = 0.001,
                     h_ip=0.1)

c.W_ie = utils.Bunch(use_sparse=False,
                     lamb=np.inf,
                     avoid_self_connections=False)

c.steps_plastic = 5000000
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

c.noise_sig = np.sqrt(0.05)
c.fast_inhibit = False
c.k_winner_take_all = False
c.ordered_thresholds = False

c.experiment.module = 'examples.zheng2013.experiment_noinput'
c.experiment.name = 'Experiment_noinput'

#######################################
c.stats.file_suffix = 'noinput'
#######################################
c.stats.rand_networks = 0

from common.sources import NoSource
source = NoSource()

# Cluster  
c.cluster.vary_param = 'with_plasticity'
c.cluster.params = [False,True]
if c.imported_mpi:
    c.cluster.NUMBER_OF_SIMS  = len(c.cluster.params)
    c.cluster.NUMBER_OF_CORES = MPI.COMM_WORLD.size
    c.cluster.NUMBER_LOCAL = c.cluster.NUMBER_OF_SIMS\
                             // c.cluster.NUMBER_OF_CORES
