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

c.W_ee = utils.Bunch(use_sparse=True,
                     lamb=0.1*c.N_e,
                     avoid_self_connections=True,
                     eta_stdp = 0.001,
                     sp_prob = 0.0,
                     sp_initial=0.000,
                     no_prune = True,
                     upper_bound = 1,
                     eta_ds = 0.1
                     )

c.W_ei = utils.Bunch(use_sparse=False,
                     lamb=np.inf,
                     avoid_self_connections=False,
                     eta_istdp = 0.000,
                     h_ip=0.1)

c.W_ie = utils.Bunch(use_sparse=False,
                     lamb=np.inf,
                     avoid_self_connections=False)


c.steps_plastic = 50000
c.steps_noplastic_train = 20000
c.steps_noplastic_test = 50000
c.N_steps = c.steps_plastic + c.steps_noplastic_train + \
                              c.steps_noplastic_test
c.display = True

c.noise_sig = 0 # 0.05 # 0.05 -> 50% #0.01 -> 10% #0.005 -> 4%

c.N_iterations = 20

c.with_plasticity = True

c.input_gain = 0.5

c.eta_ip = 0.001
h_ip_mean = float(2*c.N_u_e)/float(c.N_e)
h_ip_range = 0.01
c.h_ip = np.random.rand(c.N_e)*h_ip_range*2 + h_ip_mean - h_ip_range
c.always_ip = True

c.T_e_max = 0.5
c.T_e_min = 0.0
c.T_i_max = 1.0
c.T_i_min = 0.0
c.ordered_thresholds = True

c.fast_inhibit = True
c.k_winner_take_all = False
c.ff_inhibition = False
c.ff_inhibition_broad = 0

c.frac_A = np.array(range(11))/10.0

c.experiment.module = 'chartmann.spont.experiment_hesselmann'
c.experiment.name = 'Experiment_hesselmann'
# To check spontaneous behaviour
# when change, also change quenching (spont->train, hesselmann->test)
#~ c.experiment.module = 'chartmann.spont.experiment_spont'
#~ c.experiment.name = 'Experiment_spont'

###############################################
c.stats.file_suffix = 'ds_permuteAmbiguous'
###############################################
c.stats.save_spikes = True
c.stats.relevant_readout = False
c.stats.quenching = 'test'
c.stats.quenching_window = 2
c.stats.match = False
c.stats.lstsq_mue = 1.0
c.stats.forward_pred = 10 # Used in TrialBayes Stat
c.stats.control_rates = False
c.stats.bayes_noinput = False

from common.sources import CountingSource
c.source.prob = 0.75 # This is only here to be changed by cluster
c.source.avoid = True
c.source.N_x = 3
c.source.use_randsource = False
c.source.permute_ambiguous = True
Xes = (c.source.N_x*'X')
source = CountingSource(['A%s_'%Xes,'B%s_'%Xes],
                        np.array([[5.0/10.0, 5.0/10.0],
                                  [5.0/10.0, 5.0/10.0]]),
                        #~ np.array([[0.5,0.5],\
                                  #~ [0.5,0.5]]),\
                        c.N_u_e,c.N_u_i,c.source.avoid)
mean_word_length = np.mean([len(x) for x in source.words])
c.wait_min_plastic = 2*mean_word_length
c.wait_var_plastic = 1*mean_word_length
c.wait_min_train = 2*mean_word_length
c.wait_var_train = 1*mean_word_length
c.wait_min_test = 2*mean_word_length
c.wait_var_test = 1*mean_word_length
                        
# Cluster
c.cluster.vary_param = 'source.prob'#'with_plasticity'
if c.imported_mpi:
    c.cluster.params = np.linspace(0.1,0.9,11)#[False,True]
    c.cluster.NUMBER_OF_SIMS  = len(c.cluster.params)
    c.cluster.NUMBER_OF_CORES = MPI.COMM_WORLD.size
    c.cluster.NUMBER_LOCAL = c.cluster.NUMBER_OF_SIMS\
                             // c.cluster.NUMBER_OF_CORES
