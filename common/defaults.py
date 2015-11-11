import numpy as np
import utils
utils.backup(__file__)

'''
Default Parameters that are used when no other parameters are specified
'''

c = utils.Bunch()

# Number of units (e: excitatory, i: inhibitory, u: input)
c.N_e = 200 # if this is changed, eta_stdp has also to change
c.N_i = int(0.2*c.N_e)
c.N_u_e = int(0.05*c.N_e)
c.N_u_i = int(.6*c.N_u_e) # if c.ff_inhibition
c.N = c.N_e + c.N_i

# Each submatrix expects a Bunch with some of the following fields:
# c.use_sparse = True
# Number of connections per neuron
# c.lamb = 10 (or inf to get full connectivity)
# c.avoid_self_connections = True
# c.eta_stdp = 0.001 (or 0.0 to disable)
# c.eta_istdp = 0.001 (or 0.0 to disable)
# c.sp_prob = 0.1 (or 0 to disable)
# c.sp_initial = 0.001

c.double_synapses = False # Two excitatory synapses for each connection
c.W_ee = utils.Bunch(use_sparse=True, # internal representation
                     lamb=0.1*c.N_e, # average number of connections
                     avoid_self_connections=True,
                     # This should scale with mean weight
                     # more neurons -> smaller eta_stdp
                     # higher lamb. -> smaller eta_stdp
                     eta_stdp = 0.001, #spike-timing-dependent-plast.
                     eta_ip=0.001, # intrinsic plasticity
                     sp_prob = 0.0, # probability of insterting new con.
                     sp_initial=0.001,
                     eta_ds=0.0 # double scaling: pre- and postsyn.
                     )

c.W_ei = utils.Bunch(use_sparse=False,
                     lamb=np.inf,
                     avoid_self_connections=False,
                     eta_istdp = 0.0,
                     h_ip=0.1 # should be mean(c.h_ip) for istdp
                     )

c.W_ie = utils.Bunch(use_sparse=False,
                     lamb=np.inf,
                     avoid_self_connections=False)
                     
c.steps_plastic = 20000 # steps during self-organization
c.steps_noplastic_train = 20000 # steps for first phase w/o plasticity
c.steps_noplastic_test = 50000 # steps for 2nd phase w/o plasticity
c.N_steps = c.steps_plastic + c.steps_noplastic_train \
                            + c.steps_noplastic_test
                            
c.display = True # plotting, command-line progress, False for cluster

c.N_iterations = 20 # for averaging on cluster

c.noise_sig = 0.0 # Std of noise added at each step (mean = 0)

c.with_plasticity = True # plasticity in plastic phase

c.input_gain = 0.5 # input to each input unit when input is activated

# Intrinsic plasticity Parameters
c.eta_ip = 0.001
#~ c.h_ip = 2.0*c.N_u_e/c.N_e # Andreea heuristic
# create uniform distribution in [ip_mean-ip_range,ip_mean+ip_range]
h_ip_mean = float(2*c.N_u_e)/float(c.N_e)
h_ip_range = 0.01
c.h_ip = np.random.rand(c.N_e)*h_ip_range*2 + h_ip_mean - h_ip_range
c.always_ip = True # ip in noplastic phases
c.synaptic_scaling = True
c.inhibitory_scaling = True # Scaling of the W_ei matrix at each step

# Thresholds
c.T_e_max = 0.5
c.T_e_min = 0.0
c.T_i_max = 0.35
c.T_i_min = 0.0
c.ordered_thresholds = True # evenly spaced thresholds

# Inhibition
c.fast_inhibit = True # latest excitatory input for inhibition
c.k_winner_take_all = False
c.ff_inhibition = False # feedforward (ff) inhibition similar to ff-exc.
c.ff_inhibition_broad = 0.0 # Unspecific ff inhibition

c.experiment = utils.Bunch()
c.experiment.module = 'chartmann.spont.experiment_default' # file
c.experiment.name = 'Experiment_default' # class

c.stats = utils.Bunch() # Parameters for stats/plotting
#######################################
c.stats.file_suffix = 'default' # suffix appended to plots
#######################################
c.stats.rand_networks = 0 # control for some network measures
c.stats.save_spikes = True # memory issues
c.stats.relevant_readout = False # BayesStat: train only on readout step
c.stats.quenching = 'train' # this also affects evokedpred
c.stats.quenching_window = 2 # to each side
c.stats.match = False # EvokedPredStat: different spike representation
c.stats.lstsq_mue = 1 # regularizer of least-squares
c.stats.forward_pred = 10 # Used in TrialBayes Stat
c.stats.control_rates = False # compute rates for letters and resample
c.stats.bayes_noinput = False # exclude input units from predictions
c.stats.ISI_step = 1 # take only every ith step for ISI
# c.stats.only_last = 3000 # affects many stats: take only last x steps

c.source = utils.Bunch() # Parameters for the input source
c.source.control = False # different control sources
c.source.avoid = False # avoid self-connections

# Set here because often wordlength from source needed
# Interval and maximal variation of interval between trails
c.wait_min_plastic = 0
c.wait_var_plastic = 0
c.wait_min_train = 0
c.wait_var_train = 0
c.wait_min_test = 0
c.wait_var_test = 0

c.cluster = utils.Bunch()
c.cluster.vary_param = 'with_plasticity' # which parameter to explore
c.cluster.params = [False,True] # parameter values to explore
# Check if parallelization possible
try:
    from mpi4py import MPI
    c.imported_mpi = True
except ImportError:
    c.imported_mpi = False
if c.imported_mpi:
    # This MUST BE the number of different parameters
    c.cluster.NUMBER_OF_SIMS  = len(c.cluster.params)
    c.cluster.NUMBER_OF_CORES = MPI.COMM_WORLD.size
    c.cluster.NUMBER_LOCAL = c.cluster.NUMBER_OF_SIMS\
                             // c.cluster.NUMBER_OF_CORES
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
