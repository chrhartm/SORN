from __future__ import division
from pylab import *
import utils
utils.backup(__file__)

from chartmann.plot_single import plot_results as plot_results_single
from chartmann.plot_cluster import (plot_results 
                                    as plot_results_cluster)
from common.sources import NoSource
from common.experiments import AbstractExperiment
from common.sorn_stats import *

class Experiment_noinput(AbstractExperiment):
    def start(self):
        super(Experiment_noinput,self).start()   
        c = self.params.c
        
        self.source = NoSource()
        
        stats_all = [
                     ActivityStat(),
                     ConnectionFractionStat(),
                     WeightHistoryStat('W_ee',record_every_nth=
                                                        c.N_steps/1000),
                     ParamTrackerStat()
                    ]
        stats_single =  [
                         PopulationVariance(),
                         ISIsStat(),
                         EndWeightStat(),
                         WeightChangeStat(),
                         WeightLifetimeStat(),
                    ]
        return (self.source,stats_all+stats_single,stats_all)
        
    def reset(self,sorn):
        super(Experiment_noinput,self).reset(sorn)
        c = self.params.c
        stats = sorn.stats # init sets sorn.stats to None
        sorn.__init__(c,self.source)
        sorn.stats = stats
            
    def run(self,sorn):
        super(Experiment_noinput,self).run(sorn)
        c = self.params.c
        
        sorn.simulation(c.N_steps)
        
        return {'source_plastic':self.source}
     
    def plot_single(self,path,filename):
        plot_results_single(path,filename)
    def plot_cluster(self,path,filename):
        plot_results_cluster(path,filename)
