from __future__ import division
import matplotlib
matplotlib.use('Agg')
import os
import sys
from importlib import import_module
sys.path.insert(1,"../")

import utils
# This assumes that you are in the folder "common"
utils.initialise_backup(mount="../", dest="../backup")
utils.backup(__file__)

from utils.backup import dest_directory
from common.stats import StatsCollection
from common.sorn import Sorn
import cPickle as pickle
import gzip
import numpy as np
import matplotlib.pyplot as plt
from mpi4py import MPI
import time

comm = MPI.COMM_WORLD
rank = comm.rank

# Start debugging mode when an error is raised
def debugger(type,flag):
    print 'In debugger!'
    #~ import ipdb
    #~ ipdb.set_trace()
np.seterrcall(debugger)    
np.seterr(all='call')

# Parameters are read from the second command line argument
param = import_module(utils.param_file())
experiment_module = import_module(param.c.experiment.module)
experiment_name = param.c.experiment.name

c = param.c
if rank==0:
    logfilepath = utils.logfilename('')+'/'
else:
    logfilepath = None
c.logfilepath = comm.bcast(logfilepath, root=0)
c.display = False # Keep logfile small.

dlog = utils.DataLog()
dlog.progress("Code Backed up")

# Create one logger for all simulations
dlog.progress("NUMBER_OF_SIMS = %d"%c.cluster.NUMBER_OF_SIMS)
result_path = c.logfilepath+'result.h5'
dlog.set_handler("*", utils.StoreToH5, result_path)
dlog.append('c', utils.unbunchify(c))

# Loop over all simulations on this core
for num_local in range(c.cluster.NUMBER_LOCAL):
    dlog.progress("Outer Iteration %d of %d" %(num_local+1,
                                               c.cluster.NUMBER_LOCAL))
    if c.cluster.__contains__('params'):
        current_param = c.cluster.params[rank*
                                       c.cluster.NUMBER_LOCAL+num_local]
    else:
        # Determine current parameter by combining the rank of this core
        # = integer in range (0,#cores) and the current loop iteration
        param_range = c.cluster.param_max - c.cluster.param_min
        current_param = ((rank*c.cluster.NUMBER_LOCAL)
                        +num_local*1.0)/(c.cluster.NUMBER_OF_CORES*
                        c.cluster.NUMBER_LOCAL*1.0)*(
                        param_range+c.cluster.param_min)
    
    # replace regular parameter in c with current_param
    # the search is necessary because parameters might be split by a dot
    # TODO there should be an easier way to do this
    c_childs = c.cluster.vary_param.split('.')
    parameters = []
    tmp = c
    for child in c_childs:
        parameters.append(tmp)
        tmp = tmp[child]
    parameters[-1][c_childs[-1]] = current_param
    for i in range(np.shape(c_childs)[0]-1):
        parameters[-(i+2)][c_childs[-(i+1)]] = parameters[-(i+1)]
    c = parameters[0]
    tmp = c
    for item in c.cluster.vary_param.split('.'):
        tmp = tmp[item]
    dlog.progress("%s = %.3f"%(c.cluster.vary_param,tmp))
    # --Finished finding the parameter
    
    # Change parameters that are easily dealt with, otherwise send to
    # experiment
    experiment_param = None
    if c.cluster.vary_param == 'steps_plastic':
        c.steps_plastic = int(c.steps_plastic)
        c.N_steps = c.steps_plastic+c.steps_noplastic_train+\
                    c.steps_noplastic_test
    elif c.cluster.vary_param == 'steps_noplastic_train':
        c.steps_noplastic_train = int(c.steps_noplastic_train)
        c.N_steps = c.steps_plastic+c.steps_noplastic_train+\
                    c.steps_noplastic_test
    elif c.cluster.vary_param == 'with_plasticity':
        pass
    elif c.cluster.vary_param == 'h_ip':
        pass  
    elif c.cluster.vary_param == 'source.control':
        pass
    else:
        experiment_param = c.cluster.vary_param
    
    # Prepare experiment
    experiment = getattr(experiment_module,experiment_name)(param,
                                         cluster_param=experiment_param) 
    # Save current parameter for logging
    c.cluster.current_param = current_param
    
    # Start experiment
    (source,_,stats_cluster) = experiment.start()
    
    sorn = Sorn(c,source)
    
    # Create a StatsCollection and fill it with methods for all 
    # statistics that should be tracked (and later plotted)
    stats = StatsCollection(sorn,dlog)
    stats.methods = stats_cluster
    sorn.stats = stats
    
    # Run experiments
    experiment.reset(sorn)
    sorn.stats.start()
    dlog.progress("Start inner loop")
    for n in range(c.N_iterations):
        experiment.reset(sorn)
        stats.clear()
        
        dlog.progress("Inner Iteration %d of %d at param %.3f"%(n+1,
                                                    c.N_iterations,tmp))
                                                    
        pickle_objects = experiment.run(sorn)

        # Save sources etc
        for key in pickle_objects:
            filename = os.path.join(c.logfilepath,"%s_%s_%.3f.pickle"
                              %(key,c.cluster.vary_param,current_param))
            # using reference directly doesn't work
            topickle = pickle_objects[key]
            pickle.dump(topickle, gzip.open(filename,"wb"),
             pickle.HIGHEST_PROTOCOL)    

        if sorn.c.stats.control_rates:
            experiment.control_rates(sorn)
        
        dlog.progress("Simulation done for param %.3f"%tmp)
        sorn.stats.cluster_report(c.cluster)
        dlog.progress("Report done for param %.3f"%tmp)

dlog.close()
if rank==0:
    experiment.plot_cluster(dest_directory,
                            os.path.join('common','result.h5'))
        
