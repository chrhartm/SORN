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

class Experiment_spont(AbstractExperiment):
    def start(self):
        super(Experiment_spont,self).start()   
        c = self.params.c
        
        if self.cluster_param == 'source.prob':
            prob = c.source.prob
            assert(prob>=0 and prob<=1)
            self.params.source = CountingSource(
                                          self.params.source.words,
                                          np.array([[prob,1.0-prob],
                                                    [prob,1.0-prob]]),
                                          c.N_u_e,c.source.avoid)
        
        if c.source.use_randsource:
            self.params.source = CountingSource.init_simple(
                    c.source.N_words,
                    c.source.N_letters,c.source.word_length,
                    c.source.max_fold_prob,c.N_u_e,c.N_u_i,
                    c.source.avoid)
                    
        self.inputsource = TrialSource(self.params.source, 
                         c.wait_min_plastic, c.wait_var_plastic, 
                         zeros(self.params.source.N_a), 'reset')
        
        # Stats
        inputtrainsteps = c.steps_plastic + c.steps_noplastic_train
        # For PatternProbabilityStat
        if c.steps_noplastic_test > 6000:
            burnin = 3000
        else:
            burnin = c.steps_noplastic_test//2
        shuffled_indices = arange(c.N_e)
        np.random.shuffle(shuffled_indices)
        N_subset = 8
        start_train = c.steps_plastic+burnin
        half_train = start_train+(inputtrainsteps-start_train)//2
        start_test = inputtrainsteps+burnin
        half_test = start_test+(c.N_steps-start_test)//2
        
        stats_all = [
                     InputIndexStat(),
                     SpikesStat(),
                     InputUnitsStat(),
                     NormLastStat(),
                     SpontPatternStat(),
                     ParamTrackerStat(),
                     EvokedPredStat(
                            traintimes=[c.steps_plastic,
                                        c.steps_plastic+
                                        c.steps_noplastic_train//2],
                            testtimes =[c.steps_plastic+
                                        c.steps_noplastic_train//2,
                                        c.steps_plastic+
                                        c.steps_noplastic_train], 
                                        traintest=c.stats.quenching),
                    ]
        stats_single = [
                         ActivityStat(),
                         SpikesStat(inhibitory=True),
                         ISIsStat(interval=[start_test,c.N_steps]),
                         ConnectionFractionStat(),
                         EndWeightStat(),
                         #~ BalancedStat(), # takes lots of time and mem
                         CondProbStat(),
                         SpontIndexStat(),
                         SVDStat(),
                         SVDStat_U(),
                         SVDStat_V(),
                         SpontTransitionStat(),
                         InputUnitsStat(),
                         PatternProbabilityStat(
                                        [[start_train,half_train],
                                         [half_train,inputtrainsteps],
                                         [start_test,half_test],
                                         [half_test,c.N_steps]],
                                          shuffled_indices[:N_subset]),
                         WeightHistoryStat('W_ee',record_every_nth=100),
                         WeightHistoryStat('W_eu',
                                           record_every_nth=9999999)
                        ]
        if c.double_synapses:
            stats_single += [WeightHistoryStat('W_ee_2',
                                           record_every_nth=100)]
        return (self.inputsource,stats_all+stats_single,stats_all)
        
    def reset(self,sorn):
        super(Experiment_spont,self).reset(sorn)
        c = self.params.c
        stats = sorn.stats # init sets sorn.stats to None
        sorn.__init__(c,self.inputsource)
        sorn.stats = stats
            
    def run(self,sorn):
        super(Experiment_spont,self).run(sorn)
        c = self.params.c
        
        sorn.simulation(c.steps_plastic)
        sorn.update = False
        # Run with trials
        trialsource = TrialSource(self.inputsource.source, 
                                  c.wait_min_train, c.wait_var_train, 
                                  zeros(self.inputsource.source.N_a), 
                                  'reset')
        sorn.source = trialsource
        shuffle(sorn.x)
        shuffle(sorn.y)
        sorn.simulation(c.steps_noplastic_train)
        
        # Run with spont
        spontsource = NoSource(sorn.source.source.N_a)
        sorn.source = spontsource
        shuffle(sorn.x)
        shuffle(sorn.y)
        # Simulate spontaneous activity
        sorn.c.ff_inhibition_broad = 0
        if not c.always_ip:
            sorn.c.eta_ip = 0
        sorn.simulation(c.steps_noplastic_test)
        
        return {'source_plastic':self.inputsource,
                'source_train':trialsource,
                'source_test':spontsource}
     
    def plot_single(self,path,filename):
        plot_results_single(path,filename)
    def plot_cluster(self,path,filename):
        plot_results_cluster(path,filename)
