from __future__ import division
import utils
import numpy as np
utils.backup(__file__)


class AbstractExperiment(object):
    '''An experiment encapsulates everything experiment-specific for a 
    simulation. This is then used either for single runs or for
    multiple runs on the cluster.'''
    def __init__(self,params,cluster_param=None):
        self.params = params
        self.cluster_param = cluster_param
                
    def start(self):
        '''This is called initially to prepare the experiment. It should
        return a sorn and a list of initialized stats'''
        self.ff_inhibition_broad = self.params.c.ff_inhibition_broad
        self.eta_ip = self.params.c.eta_ip

    def reset(self,sorn):
        '''This is called at every iteration on the cluster.'''
        sorn.update = self.params.c.with_plasticity
        if not self.params.c.with_plasticity:
            sorn.c.eta_ip = 0
        else:
            sorn.c.eta_ip = self.eta_ip
        sorn.c.ff_inhibition_broad = self.ff_inhibition_broad

    def run(self,sorn):
        '''Run Forest, run! It should return a dictionary of objects to 
        pickle (name, obj) including the sorn'''
        pass
        
    def plot_single(self,path,filename):
        pass
        
    def plot_cluster(self,path,filename):
        pass
        
    def control_rates(self,sorn):
        # Control: Firing-rate model: Substitute spikes by drawing 
        # random spikes according to firing rate for each inputindex
        inputtrainsteps = sorn.c.steps_plastic+\
                          sorn.c.steps_noplastic_train
        intervals = [(0,sorn.c.steps_plastic),
                     (sorn.c.steps_plastic,inputtrainsteps),
                     (inputtrainsteps,inputtrainsteps+
                      sorn.c.steps_noplastic_test)]
        for (start,stop) in intervals:
            indices = set(sorn.stats.c.inputindex[start:stop])
            for i in indices:
                ii = np.where(sorn.stats.c.inputindex[start:stop]==i)[0]
                # The [:,None] at the end adds an extra dimension 
                # for the later comparison
                mean_spikes = np.mean(
                      sorn.stats.c.spikes[:,start:stop][:,ii],1)[:,None]
                sorn.stats.c.spikes[:,start:stop][:,ii] = \
                          np.random.rand(sorn.c.N_e,len(ii))<mean_spikes
