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

class Experiment_hesselmann(AbstractExperiment):
    def start(self):
        super(Experiment_hesselmann,self).start()   
        c = self.params.c
        
        if self.cluster_param == 'source.prob' and c.imported_mpi:
            prob = c.source.prob
            assert(prob>=0 and prob<=1)
            self.params.source = CountingSource(
                                          self.params.source.words,
                                          np.array([[prob,1.0-prob],
                                                    [prob,1.0-prob]]),
                                          c.N_u_e,c.source.avoid)
        
        self.source = TrialSource(self.params.source, 
                         c.wait_min_plastic, c.wait_var_plastic, 
                         zeros(self.params.source.N_a), 'reset')
                         
        self.trainsource = copy.deepcopy(self.source)
        self.trainsource.blank_min_length = c.wait_min_train
        self.trainsource.blank_var_length = c.wait_var_train
        
        
        # Stats
        inputtrainsteps = c.steps_plastic + c.steps_noplastic_train
        if c.steps_noplastic_test < 100000:
            pred_test = c.steps_noplastic_test // 2
        else:
            pred_test = 50000
        if c.stats.quenching == 'train':
            evoked_train = [c.steps_plastic,
                            c.steps_plastic+c.steps_noplastic_train//2]
            evoked_test = [c.steps_plastic+c.steps_noplastic_train//2,
                           c.steps_plastic+c.steps_noplastic_train]
        elif c.stats.quenching == 'test':
            evoked_train = [inputtrainsteps,-pred_test]
            evoked_test = [-pred_test,-1]
            
        stats_all = [
                     InputIndexStat(),
                     SpikesStat(),
                     ParamTrackerStat(),
                     BayesStat(pred_pos=0),
                     #~ AttractorDynamicsStat(),
                     HistoryStat(var='W_eu.W',collection='gather',
                                 record_every_nth=100000000),
                     TrialBayesStat(),
                     SpontBayesStat(),
                     OutputDistStat(),
                     EvokedPredStat(traintimes = evoked_train,
                                    testtimes = evoked_test,
                                    traintest = c.stats.quenching),
                     InputUnitsStat(),
                     ]
        
        stats_single = [
                     ActivityStat(),
                     WeightHistoryStat('W_ee',record_every_nth=100),
                     WeightHistoryStat('W_eu',
                                           record_every_nth=9999999),
                     ConnectionFractionStat(),
                     ISIsStat(),
                     SpikesStat(inhibitory=True),
                     CondProbStat(),
                     EndWeightStat(),
                     #~ BalancedStat(), # Takes forever
                     #~ RateStat(),
                     NormLastStat(),
                     #~ SVDStat(),
                     #~ SVDStat_U(),
                     #~ SVDStat_V(),
                     ]
                 
        return (self.source,stats_all+stats_single,stats_all)
        
    def reset(self,sorn):
        super(Experiment_hesselmann,self).reset(sorn)
        c = self.params.c
        stats = sorn.stats # init sets sorn.stats to None
        sorn.__init__(c,self.source)
        sorn.stats = stats
            
    def run(self,sorn):
        super(Experiment_hesselmann,self).run(sorn)
        c = self.params.c
        
        sorn.simulation(c.steps_plastic)
        sorn.update = False
        
        ## Generate Training and test data
        # Andreea also did this
        sorn.x = np.zeros(sorn.c.N_e)
        sorn.y = np.zeros(sorn.c.N_i)
        sorn.source = self.trainsource
        # Generate Training Data
        sorn.simulation(c.steps_noplastic_train)
        sorn.x = np.zeros(sorn.c.N_e)
        sorn.y = np.zeros(sorn.c.N_i)

        # Generate new source with mixed letters
        # Save old mappings
        old_W_eu = sorn.W_eu.W
        source = self.source.source
        A_neurons = where(sorn.W_eu.W[:,source.lookup['A']]==1)[0]
        B_neurons = where(sorn.W_eu.W[:,source.lookup['B']]==1)[0]
        X_neurons = where(sorn.W_eu.W[:,source.lookup['X']]==1)[0]
        # First generate source with len(frac_A) words of equal prob
        N_words = len(c.frac_A)
        letters = "CDEFGHIJKL" 
        Xes = (sorn.c.source.N_x*'X')
        word_list = ['A%s_'%Xes,'B%s_'%Xes]                               
        frac_A_letters = ['B']
        for i in range(N_words-2):
            word_list.append('%s%s_'%(letters[i],Xes))                     
            frac_A_letters.append(letters[i])
        frac_A_letters.append('A')
        
        probs = np.ones((N_words,N_words))*(1.0/N_words)
        source_new = CountingSource(word_list,probs,c.N_u_e,True,
                           permute_ambiguous=c.source.permute_ambiguous)
        # Then take weight matrices of these guys and set them according
        # to frac_A
        new_W_eu = source_new.generate_connection_e(c.N_e)
        new_W_eu.W *= 0
        new_W_eu.W[A_neurons,source_new.lookup['A']] = 1
        new_W_eu.W[B_neurons,source_new.lookup['B']] = 1
        new_W_eu.W[X_neurons,source_new.lookup['X']] = 1
        # from 1 to len-1 because 0 and 1 already included in fracs
        for i in range(1,len(frac_A_letters)-1):
            new_neurons = hstack(
                          (A_neurons[:int(len(A_neurons)*c.frac_A[i])],
                           B_neurons[int(len(A_neurons)*c.frac_A[i]):]))
            new_W_eu.W[new_neurons,
                       source_new.lookup[frac_A_letters[i]]] = 1
            
        # Assign the source and new Matrix to SORN
        sorn.W_eu = new_W_eu
        sorn.W_iu = source_new.generate_connection_i(c.N_i)
        trialsource_new = TrialSource(source_new, c.wait_min_test,
                                c.wait_var_test, zeros(source_new.N_a), 
                                'reset')
        sorn.source = trialsource_new
        
        sorn.simulation(c.steps_noplastic_test)
        
        return {'source_plastic':self.source,
                'source_train':self.trainsource,
                'source_test':trialsource_new}
     
    def plot_single(self,path,filename):
        plot_results_single(path,filename)
    def plot_cluster(self,path,filename):
        plot_results_cluster(path,filename)
