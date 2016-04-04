from __future__ import division
from pylab import *
import utils
utils.backup(__file__)

from chartmann.plot_single import plot_results as plot_results_single
from chartmann.plot_cluster import (plot_results
                                    as plot_results_cluster)
from common.sources import CountingSource, TrialSource, NoSource
from common.experiments import AbstractExperiment
from common.sorn_stats import *
import copy

class Experiment_sequence(AbstractExperiment):
    def start(self):
        super(Experiment_sequence,self).start()   
        c = self.params.c
        # Create paper-specific sources
        self.test_words = c.source.test_words
        if not c.source.control:
            source = CountingSource(['ABCD'],np.array([[1.]]),
                           c.N_u_e,c.N_u_i,c.source.avoid)
        else:
            from itertools import permutations
            source = CountingSource.init_simple(24,4,[4,4],1,
                           c.N_u_e,c.N_u_i,c.source.avoid, words = 
                           [''.join(x) for x in (permutations('ABCD'))])
        # Already add letters for later
        source.alphabet = unique("".join(source.words)+'E_')
        source.N_a = len(source.alphabet)
        source.lookup = dict(zip(source.alphabet,\
                                       range(source.N_a)))

        source = TrialSource(source, c.wait_min_plastic, 
                       c.wait_var_plastic,zeros(source.N_a),'reset')
        self.source_archived = copy.deepcopy(source)

        inputtrainsteps = c.steps_plastic + c.steps_noplastic_train

        stats_single = [
                         ActivityStat(),
                         InputIndexStat(),
                         SpikesStat(),
                         ISIsStat(interval=[0, c.steps_plastic]),
                         ConnectionFractionStat(),
                        ]
        stats_all = [
                     ParamTrackerStat(),
                     EndWeightStat(),
                     InputUnitsStat(),
                     MeanActivityStat(start=inputtrainsteps,
                      stop=c.N_steps,
                      N_indices=len(''.join((self.test_words)))+1,
                                    LFP=False),
                     MeanPatternStat(start=c.steps_plastic,
                      stop=c.N_steps,
                      N_indices=len(''.join((self.test_words)))+1)
                    ]
        
        return (source,stats_single+stats_all,stats_all)
        
    def reset(self,sorn):
        super(Experiment_sequence,self).reset(sorn)
        c = self.params.c
        source = copy.deepcopy(self.source_archived)
        stats = sorn.stats # init sets sorn.stats to None
        sorn.__init__(c,source)
        sorn.stats = stats
            
    def run(self,sorn):
        super(Experiment_sequence,self).run(sorn)
        c = self.params.c
        
        # Simulate with plasticity
        sorn.simulation(c.steps_plastic)
        
        # Turn off plasticity
        sorn.update = False
        
        # Shuffle (Gordon's idea)
        shuffle(sorn.x)
        shuffle(sorn.y)
        
        # Run with test words
        source = sorn.source.source
        source.words = self.test_words
        N_words = len(source.words)
        source.probs = array([ones(N_words)]*N_words)
        source.probs /= sum(source.probs,1)
        source.glob_ind = [0]
        source.glob_ind.extend(cumsum(map(len,source.words)))
        source.word_index = 0
        source.ind = 0
                
        spontsource = NoSource(sorn.source.source.N_a)
        sorn.source = spontsource
        if not c.always_ip:
            sorn.c.eta_ip = 0
        sorn.simulation(c.steps_noplastic_train)
        # Run again
        trialsource = TrialSource(source, c.wait_min_train, 
                       c.wait_var_train, zeros(source.N_a), 'reset')
        sorn.source = trialsource
        sorn.simulation(c.steps_noplastic_test)
        
        return {'source_plastic':self.source_archived,
                 'source_train':sorn.source,
                 'source_test':sorn.source}
     
    def plot_single(self,path,filename):
        plot_results_single(path,filename)
    def plot_cluster(self,path,filename):
        plot_results_cluster(path,filename)
