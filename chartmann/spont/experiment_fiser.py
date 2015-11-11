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

class Experiment_fiser(AbstractExperiment):
    def start(self):
        super(Experiment_fiser,self).start()   
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
 
        controlsource = copy.deepcopy(self.params.source)
        controlsource.words = [x[::-1] for x in 
                               self.params.source.words]
                               
        # Control with single letters
        #~ controlsource.words = controlsource.alphabet
        #~ N_words = len(controlsource.words)
        #~ probs = array([ones(N_words)]*N_words)
        #~ controlsource.probs = probs/sum(probs,1)
        #~ controlsource.glob_ind = [0]
        #~ controlsource.glob_ind.extend(cumsum(map(len,
                                            #~ controlsource.words)))
        
        # Control with different words
        # Check if ABCD source
        if controlsource.words == ['DCBA','HGFE']:
            controlsource.words = ['EDCBA','HGF']
                    
        self.plasticsource = TrialSource(self.params.source, 
                                c.wait_min_plastic, c.wait_var_plastic, 
                                zeros(self.params.source.N_a), 'reset')
        self.controlsource = TrialSource(controlsource,
                                c.wait_min_train, c.wait_var_train,
                                zeros(controlsource.N_a), 'reset')
        self.spontsource = NoSource(controlsource.N_a)
                
        #Stats
        inputtrainsteps = c.steps_plastic + c.steps_noplastic_train
        # For PatternProbabilityStat
        if c.steps_noplastic_test > 10000:
            burnin = 5000
        else:
            burnin = c.steps_noplastic_test//2
        shuffled_indices = arange(c.N_e)
        np.random.shuffle(shuffled_indices)
        N_subset = 16
        start_train = c.steps_plastic+burnin
        half_train = start_train+(inputtrainsteps-start_train)//2
        start_test = inputtrainsteps+burnin
        half_test = start_test+(c.N_steps-start_test)//2
        # The output dimensions of these stats have to be independent
        # of the number of steps!
        stats_all = [                         
                     ParamTrackerStat(),
                     PatternProbabilityStat(
                                    [[start_train,half_train],
                                     [half_train+burnin,inputtrainsteps],
                                     [start_test,half_test],
                                     [half_test,c.N_steps]],
                                      shuffled_indices[:N_subset],
                                      zero_correction=True)
                    ]
        stats_single = [      
                         InputIndexStat(),
                         SpikesStat(),
                         InputUnitsStat(),
                         ActivityStat(),
                         SpikesStat(inhibitory=True),
                         ISIsStat(interval=[c.steps_plastic,
                                            inputtrainsteps]),
                         ConnectionFractionStat(),
                         InputUnitsStat(),
                        ]
                        
        return (self.plasticsource,stats_single+stats_all,stats_all)
        
    def reset(self,sorn):
        super(Experiment_fiser,self).reset(sorn)
        c = self.params.c
        source = self.plasticsource
        stats = sorn.stats # init sets sorn.stats to None
        sorn.__init__(c,source)
        sorn.stats = stats
            
    def run(self,sorn):
        super(Experiment_fiser,self).run(sorn)
        c = self.params.c
        
        # Simulate with plasticity
        sorn.simulation(c.steps_plastic)
        
        # Turn off plasticity
        sorn.update = False
        if not c.always_ip:
            sorn.c.eta_ip = 0
        
        # Shuffle (Gordon's idea)
        shuffle(sorn.x)
        shuffle(sorn.y)
        
        sorn.simulation(c.steps_noplastic_train//2)
        shuffle(sorn.x)
        shuffle(sorn.y)

        # Run with control source
        if c.source.control:
            sorn.source = self.controlsource
            trainsource = self.controlsource
        else:
            trainsource = self.plasticsource        

        sorn.simulation(c.steps_noplastic_train
                        -c.steps_noplastic_train//2)
        # Run without input
        sorn.source = self.spontsource
        sorn.c.ff_inhibition_broad = 0
        sorn.simulation(c.steps_noplastic_test)
        
        return {'source_plastic':self.plasticsource,
                'source_train':trainsource,
                'source_test':self.spontsource}
     
    def plot_single(self,path,filename):
        plot_results_single(path,filename)
    def plot_cluster(self,path,filename):
        plot_results_cluster(path,filename)
