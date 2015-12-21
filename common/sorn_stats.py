from __future__ import division
from pylab import *

import utils
utils.backup(__file__)

from stats import AbstractStat
from stats import HistoryStat
from stats import _getvar
from common.sources import TrialSource
from utils.lstsq_reg import lstsq_reg

import cPickle as pickle
import gzip

def load_source(name,c):
    try:
        filename = c.logfilepath+name+".pickle"
        sourcefile = gzip.open(filename,"r")
    except IOError: # Cluster
        filename = c.logfilepath+\
                   name+"_%s_%.3f.pickle"\
                   %(c.cluster.vary_param,\
                     c.cluster.current_param)
        sourcefile = gzip.open(filename,"r")
    source = pickle.load(sourcefile)
    
    if isinstance(source,TrialSource):
        source = source.source
    return source

class CounterStat(AbstractStat):
    def __init__(self):
        self.name = 'num_steps'
        self.collection = "reduce"
    def start(self,c,obj):
        c[self.name] = 0.0 # Everything needs to be a float :-/
    def add(self,c,obj):
        c[self.name] += 1
    def report(self,c,obj):
        return array(c[self.name]) # And an array :-/

# By making CounterStat a little longer we can make ClearCounterStat a 
# lot shorter
class ClearCounterStat(CounterStat):
    def __init__(self):
        self.name = 'counter'
        self.collection = "ignore"
        (self.clear,self.start) = (self.start,self.clear)

class PopulationVariance(AbstractStat):
    def __init__(self):
        self.name = 'pop_var'
        self.collection = 'reduce'
    def clear(self,c,obj):
        N = obj.c.N_e
        c.pop_var = zeros(N+1)
    def add(self,c,obj):
        n = sum(obj.x)
        c.pop_var[n] += 1.0
    def report(self,c,obj):
        return c.pop_var

class ActivityStat(AbstractStat):
    """
    Gathers the state of the network at each step
    
    If the parameter only_last is set, only the first and last steps are
    collected
    """
    def __init__(self):
        self.name = 'activity'
        self.collection = 'gather'
    def clear(self,c,sorn):
        if sorn.c.stats.has_key('only_last'):
            c.activity = zeros(sorn.c.stats.only_last\
                                 +sorn.c.stats.only_last)
        else:
            c.activity = zeros(sorn.c.N_steps)
        self.step = 0
    def add(self,c,sorn):
        if sorn.c.stats.has_key('only_last'):
            new_step = self.step - (sorn.c.N_steps\
                                    -sorn.c.stats.only_last)
            if new_step >= 0:
                c.activity[new_step+sorn.c.stats.only_last] \
                    = sum(sorn.x)/sorn.c.N_e
            elif self.step % (sorn.c.N_steps\
                              //sorn.c.stats.only_last) == 0:
                c.activity[self.step//(sorn.c.N_steps\
                    //sorn.c.stats.only_last)] = sum(sorn.x)/sorn.c.N_e
        else:
            c.activity[self.step] = sum(sorn.x)/sorn.c.N_e
        self.step += 1
    def report(self,c,sorn):
        return c.activity
        
class InputIndexStat(AbstractStat):
    """
    Gathers the index of the input at each step
    """
    def __init__(self):
        self.name = 'InputIndex'
        self.collection = 'gather'
    def clear(self,c,sorn):
        if sorn.c.stats.has_key('only_last'):
            c.inputindex = zeros(sorn.c.stats.only_last\
                                 +sorn.c.stats.only_last)
        else:
            c.inputindex = zeros(sorn.c.N_steps)
        self.step = 0
    def add(self,c,sorn):
        if sorn.c.stats.has_key('only_last'):
            new_step = self.step - (sorn.c.N_steps\
                                    -sorn.c.stats.only_last)
            if new_step >= 0:
                c.inputindex[new_step+sorn.c.stats.only_last] \
                    = sorn.source.global_index()
            elif self.step % (sorn.c.N_steps\
                              //sorn.c.stats.only_last) == 0:
                c.inputindex[self.step//(sorn.c.N_steps\
                 //sorn.c.stats.only_last)] = sorn.source.global_index()
        else:
            c.inputindex[self.step] = sorn.source.global_index()
        self.step += 1
    def report(self,c,sorn):
        return c.inputindex
        
class WordListStat(AbstractStat):
    # OLD! use pickle of source instead!
    def __init__(self):
        self.name = 'WordList'
        self.collection = 'gather'
    def report(self,c,sorn):
        return sorn.c.words
        
class InputUnitsStat(AbstractStat):
    def __init__(self):
        self.name = 'InputUnits'
        self.collection = 'gather'
    def report(self,c,sorn):
        input_units = where(sum(sorn.W_eu.get_synapses(),1)>0)[0]
        # to make them equal in size
        tmp = array([z in input_units for z in arange(sorn.c.N_e)]) 
        return tmp+0 # cast as double
        
class NormLastStat(AbstractStat):
    '''
    This is a helper Stat that computes the normalized last spikes 
    and input indices
    '''
    def __init__(self):
        self.name = 'NormLast'
        self.collection = 'gather'
    def report(self,c,sorn):
        steps_plastic = sorn.c.steps_plastic
        steps_noplastic_train = sorn.c.steps_noplastic_train
        steps_noplastic_test = sorn.c.steps_noplastic_test
        plastic_train = steps_plastic+steps_noplastic_train
        input_spikes = c.spikes[:,steps_plastic:plastic_train]
        input_index = c.inputindex[steps_plastic:plastic_train]

        # Filter out empty states
        input_spikes = input_spikes[:,input_index != -1]
        input_index = input_index[input_index != -1]
        
        if sorn.c.stats.has_key('only_last'):
            N_comparison = sorn.c.stats.only_last
        else:
            N_comparison = 2500
        assert(N_comparison > 0)
        assert(N_comparison <= steps_noplastic_test \
             and N_comparison <= steps_noplastic_train)
        maxindex = int(max(input_index))
        
        # Only use spikes that occured at the end of learning and spont
        last_input_spikes = input_spikes[:,-N_comparison:]
        last_input_index = input_index[-N_comparison:]

        # Get the minimal occurence of an index in the last steps
        min_letter_count = inf
        for i in range(maxindex+1):
            tmp = sum(last_input_index == i)
            if min_letter_count > tmp:
                min_letter_count = tmp

        # For each index, take the same number of states from the
        # end phase of learning to avoid a bias in comparing states
        norm_last_input_spikes = np.zeros((shape(last_input_spikes)[0],\
                                    min_letter_count*(maxindex+1)))
        norm_last_input_index = np.zeros(min_letter_count*(maxindex+1))

        for i in range(maxindex+1):
            indices = find(last_input_index == i)
            norm_last_input_spikes[:,min_letter_count*i\
                                     : min_letter_count*(i+1)]\
                = last_input_spikes[:, indices[-min_letter_count:]]
            norm_last_input_index[min_letter_count*i\
                                     : min_letter_count*(i+1)]\
                = last_input_index[indices[-min_letter_count:]]

        # Shuffle to avoid argmin-problem of selecting only first match
        indices = arange(shape(norm_last_input_index)[0])
        shuffle(indices)
        norm_last_input_index = norm_last_input_index[indices]
        norm_last_input_spikes = norm_last_input_spikes[:,indices]
        c.norm_last_input_index = norm_last_input_index
        c.norm_last_input_spikes = norm_last_input_spikes   
        c.maxindex = maxindex  
        c.N_comparison = N_comparison
        to_return = array([float(N_comparison)])
        return to_return
        
class SpontPatternStat(AbstractStat):
    """
    Computes the frequency of each pattern in the spontaneous activity
    """
    def __init__(self):
        self.name = 'SpontPattern'
        self.collection = 'gather'
    def report(self,c,sorn):
        source_plastic = load_source("source_plastic",sorn.c)
        steps_noplastic_test = sorn.c.steps_noplastic_test
        spont_spikes = c.spikes[:,-steps_noplastic_test:]
        norm_last_input_index = c.norm_last_input_index
        norm_last_input_spikes = c.norm_last_input_spikes
        maxindex = c.maxindex
        N_comparison = c.N_comparison
        
        last_spont_spikes = spont_spikes[:,-N_comparison:]
                
        # Remove silent periods from spontspikes
        last_spont_spikes = last_spont_spikes[:,sum(last_spont_spikes,0)>0]
        N_comp_spont = shape(last_spont_spikes)[1]
        
        # Find for each spontaneous state the evoked state with the
        # smallest hamming distance and store the corresponding index
        similar_input = zeros(N_comp_spont)
        for i in xrange(N_comp_spont):
            most_similar = argmin(sum(abs(norm_last_input_spikes.T\
                                -last_spont_spikes[:,i]),axis=1))
            similar_input[i] = norm_last_input_index[most_similar]
        # Count the number of spontaneous states for each index and plot
        index = range(maxindex+1)
        if self.collection == 'gatherv':
            adding = 2
        else:
            adding = 1
        pattern_freqs = zeros((2,maxindex+adding))
        barcolor = []
        for i in index:
            pattern_freqs[0,i] = sum(similar_input==index[i])

        # Compare patterns
        # Forward patterns ([0,1,2,3],[4,5,6,7],...)
        patterns = array([arange(len(w))+source_plastic.glob_ind[i] \
                         for (i,w) in enumerate(source_plastic.words)])
        rev_patterns = array([x[::-1] for x in patterns])
        maxlen = max([len(x) for x in patterns])
        # Also get the reversed patterns
        if maxlen>1: # Single letters can't be reversed
            allpatterns = array(patterns.tolist()+rev_patterns.tolist())
        else:
            allpatterns = array(patterns.tolist())
        for (i,p) in enumerate(allpatterns):
            patternlen = len(p)
            for j in xrange(N_comp_spont-maxlen):
                if all(similar_input[j:j+patternlen] == p):
                    pattern_freqs[1,i] += 1
        # Marker for end of freqs
        if self.collection == 'gatherv':
            pattern_freqs[:,-1] = -1
        c.similar_input = similar_input
        return(pattern_freqs)
        
class SpontTransitionStat(AbstractStat):
    def __init__(self):
        self.name = 'SpontTransition'
        self.collection = 'gather'
    def report(self,c,sorn):
        similar_input = c.similar_input # from SpontPatternStat
        maxindex = c.maxindex
        transitions = np.zeros((maxindex+1,maxindex+1))
        for (i_from, i_to) in zip(similar_input[:-1],similar_input[1:]):
            transitions[i_to,i_from] += 1
        return transitions
        
class SpontIndexStat(AbstractStat):
    def __init__(self):
        self.name = 'SpontIndex'
        self.collection = 'gather'
    def report (self,c,sorn):
        return c.similar_input
        
class BayesStat(AbstractStat):
    def __init__(self,pred_pos = 0):
        self.name = 'Bayes'
        self.collection = 'gather'
        self.pred_pos = pred_pos # steps before M/N
    def clear(self,c,sorn):
        pass
        # If raw_prediction is input to M/N neurons, this is needed
        #~ self.M_neurons = where(sorn.W_eu.W[:,
                              #~ sorn.source.source.lookup['M']]==1)[0]
        #~ self.N_neurons = where(sorn.W_eu.W[:,
                              #~ sorn.source.source.lookup['N']]==1)[0]
    def report(self,c,sorn):
        ### Prepare spike train matrices for training and testing
        # Separate training and test data according to steps
        source_plastic = load_source("source_plastic",sorn.c)
        steps_plastic = sorn.c.steps_plastic
        N_train_steps = sorn.c.steps_noplastic_train
        N_inputtrain_steps = steps_plastic + N_train_steps
        N_test_steps = sorn.c.steps_noplastic_test
        burnin = 3000
        # Transpose because this is the way they are in test_bayes.py
        Xtrain = c.spikes[:,steps_plastic+burnin:N_inputtrain_steps].T
        Xtest = c.spikes[:,N_inputtrain_steps:].T
        assert(shape(Xtest)[0] == N_test_steps)
        inputi_train = c.inputindex[steps_plastic+burnin
                                    :N_inputtrain_steps]
        assert(shape(Xtrain)[0] == shape(inputi_train)[0])
        inputi_test = c.inputindex[N_inputtrain_steps:]
        assert(shape(inputi_test)[0]== N_test_steps)
        N_fracs = len(sorn.c.frac_A)
        
        # Filter out empty states
        if isinstance(sorn.source,TrialSource): # if TrialSource
            source = sorn.source.source
        else:
            source = sorn.source
        Xtrain = Xtrain[inputi_train != -1,:]
        inputi_train = inputi_train[inputi_train != -1]
        Xtest = Xtest[inputi_test != -1,:]
        inputi_test = inputi_test[inputi_test != -1]

        # Following snipplet modified from sorn_stats spont_stat
        # Get the minimal occurence of an index in the last steps
        maxindex = int(max(inputi_train))
        min_letter_count = inf
        for i in range(maxindex+1):
            tmp = sum(inputi_train == i)
            if min_letter_count > tmp:
                min_letter_count = tmp

        # For each index, take the same number of states from the
        # end phase of learning to avoid a bias in comparing states
        norm_Xtrain = np.zeros((min_letter_count*(maxindex+1),
                                                      shape(Xtrain)[1]))
        norm_inputi_train = np.zeros(min_letter_count*(maxindex+1))
        for i in range(maxindex+1):
            indices = find(inputi_train == i)
            norm_Xtrain[min_letter_count*i
                                     : min_letter_count*(i+1), :]\
                = Xtrain[indices[-min_letter_count:],:]
            norm_inputi_train[min_letter_count*i
                                     : min_letter_count*(i+1)]\
                = inputi_train[indices[-min_letter_count:]]
        Xtrain = norm_Xtrain
        inputi_train = norm_inputi_train
        noinput_units = where(sum(sorn.W_eu.W,1)==0)[0]
        if sorn.c.stats.bayes_noinput:
            Xtrain_noinput = Xtrain[:,noinput_units]
            Xtest_noinput = Xtest[:,noinput_units]
        else:
            Xtrain_noinput = Xtrain
            Xtest_noinput = Xtest

        assert(source_plastic.words[0][0]=="A" and
               source_plastic.words[1][0]=="B")
        A_index = source_plastic.glob_ind[0] # start of first word
        B_index = source_plastic.glob_ind[1] # start of second word
        # position from which to predict end of word
        pred_pos = len(source_plastic.words[0])-1-self.pred_pos 
        assert(pred_pos>=0 
               and pred_pos <= source_plastic.global_range())
 
        R = np.zeros((2,shape(inputi_train)[0]))    
        R[0,:] = inputi_train == A_index+pred_pos
        R[1,:] = inputi_train == B_index+pred_pos
        
        if sorn.c.stats.relevant_readout:
            Xtrain_relevant = Xtrain_noinput[((inputi_train == 
                                               A_index+pred_pos) + 
                                (inputi_train == B_index+pred_pos))>0,:]
            R_relevant = R[:,((inputi_train == A_index+pred_pos) + 
                                (inputi_train == B_index+pred_pos))>0]
            classifier = lstsq_reg(Xtrain_relevant,R_relevant.T,
                                                 sorn.c.stats.lstsq_mue)
                                
        else:
            classifier = lstsq_reg(Xtrain_noinput,R.T,
                                                 sorn.c.stats.lstsq_mue)
        
        #~ # No real difference between LogReg, BayesRidge and my thing
        #~ # If you do this, comment out raw_predictions further down
        #~ from sklearn import linear_model
        #~ clf0 = linear_model.LogisticRegression(C=1)#BayesianRidge()
        #~ clf1 = linear_model.LogisticRegression(C=1)#BayesianRidge()
        #~ clf0.fit(Xtrain_noinput,R.T[:,0])
        #~ clf1.fit(Xtrain_noinput,R.T[:,1])
        #~ raw_predictions = vstack((clf0.predict_proba(Xtest_noinput)[:,1]
                            #~ ,clf1.predict_proba(Xtest_noinput)[:,1])).T
       
        # predict 
        #~ raw_predictions = Xtest.dot(classifier)
        
        #~ # comment this out if you use sklearn
        raw_predictions = Xtest_noinput.dot(classifier)
        
        #~ # Historical stuff
        #~ # Raw predictions = total synaptic input to M/N neurons
        #~ raw_predictions[1:,0] = sum((sorn.W_ee*Xtest[:-1].T)[
                                                #~ self.M_neurons],0)
        #~ raw_predictions[1:,1] = sum((sorn.W_ee*Xtest[:-1].T)[
                                                #~ self.N_neurons],0)
        #~ # Raw predictions = total activation of M/N neurons
        #~ raw_predictions[:,0] = sum(Xtest.T[self.M_neurons],0)
        #~ raw_predictions[:,1] = sum(Xtest.T[self.N_neurons],0)
        #~ # for testing: sum(raw_predictions[indices,0])>indices+-1,2,3

        letters_for_frac = ['B']
        # Because alphabet is sorted alphabetically, this list will
        # have the letters corresponding to the list frac_A
        for l in source.alphabet:
            if not ((l=='A') or (l=='B') or (l=='M') or (l=='N') 
                     or (l=='X') or (l=='_')):
                letters_for_frac.append(l)
        letters_for_frac.append('A')
        
        output_drive = np.zeros((N_fracs,2))
        output_std = np.zeros((N_fracs,2))
        decisions = np.zeros((N_fracs,2))
        denom = np.zeros(N_fracs)
        for (s_word,s_index) in zip(source.words,source.glob_ind):
            i = ''.join(letters_for_frac).find(s_word[0])
            indices = find(inputi_test==s_index+pred_pos)
            # A predicted
            output_drive[i,0] += mean(raw_predictions[indices,0]) 
            # B predicted
            output_drive[i,1] += mean(raw_predictions[indices,1]) 
            decisions[i,0]    += mean(raw_predictions[indices,0]>\
                                        raw_predictions[indices,1])
            decisions[i,1]    += mean(raw_predictions[indices,1]>=\
                                        raw_predictions[indices,0])
            output_std[i,0]   +=  std(raw_predictions[indices,0])
            output_std[i,1]   +=  std(raw_predictions[indices,1])
            denom[i]          += 1
        # Some words occur more than once
        output_drive[:,0] /= denom
        output_drive[:,1] /= denom
        output_std[:,0]   /= denom
        output_std[:,1]   /= denom
        decisions[:,0] /= denom
        decisions[:,1] /= denom
        
        # for other stats (e.g. SpontBayesStat)
        c.pred_pos = pred_pos
        c.Xtest = Xtest
        c.raw_predictions = raw_predictions
        c.inputi_test = inputi_test
        c.letters_for_frac = letters_for_frac
        c.classifier = classifier
        c.noinput_units = noinput_units
            
        to_return = hstack((output_drive,output_std,decisions))
        
        return to_return
        
class AttractorDynamicsStat(AbstractStat):
    """
    This stat tracks the distance between output gains during the 
    input presentation to determine whether the decision is based on
    attractor dynamics
    """
    def __init__(self):
        self.name = 'AttractorDynamics'
        self.collection = 'gather'
    def report(self,c,sorn):
        # Read stuff in
        letters_for_frac = c.letters_for_frac
        if isinstance(sorn.source,TrialSource): # if TrialSource
            source = sorn.source.source
        else:
            source = sorn.source
        word_length = min([len(x) for x in source.words])
        N_words = len(source.words)
        N_fracs = len(sorn.c.frac_A)
        bayes_stat = None
        for stat in sorn.stats.methods:
            if stat.name is 'Bayes':
                bayes_stat = stat
                break
        assert(bayes_stat is not None)
        pred_pos_old = bayes_stat.pred_pos
        
        #output_dist = np.zeros((word_length-1,N_fracs))
        output_dist = np.zeros((word_length,N_fracs))
        min_trials = inf
        for i in range(int(max(c.inputi_test))+1):
            tmp = sum(c.inputi_test == i)
            if min_trials > tmp:
                min_trials = tmp
        decisions = np.zeros((N_words,word_length,min_trials),\
                                                        dtype=np.bool)
        seq_count = np.zeros((N_words,4))
        
        
        for (p,pp) in enumerate(arange(0,word_length)):
            bayes_stat.pred_pos = pp
            bayes_stat.report(c,sorn)
            pred_pos = c.pred_pos
            raw_predictions = c.raw_predictions
            inputi_test = c.inputi_test
            #~ summed = abs(raw_predictions[:,0])+abs(raw_predictions[:,1])
            #~ summed[summed<1e-10] = 1 # if predicted 0, leave at 0
            #~ raw_predictions[:,0] /= summed
            #~ raw_predictions[:,1] /= summed

            denom = np.zeros((N_fracs))
            for (w,(s_word,s_index)) in enumerate(zip(source.words,
                                                      source.glob_ind)):
                i = ''.join(letters_for_frac).find(s_word[0])
                indices = find(inputi_test==s_index+pred_pos)
                tmp = abs(raw_predictions[indices,0]-
                          raw_predictions[indices,1])
                output_dist[p,i] += mean(tmp)
                decisions[w,p,:] = raw_predictions[
                                indices[-min_trials:],0]>\
                                raw_predictions[indices[-min_trials:],1]
                denom[i]          += 1
            output_dist[p,:] /= denom
        
        for i in range(N_words):
            # Full-length 1s to be expected
            seq_count[i,0] = ((sum(decisions[i])/(1.*min_trials*
                                word_length))**(word_length))*min_trials
            # Actual 1-series
            seq_count[i,1] = sum(sum(decisions[i],0)==word_length)
            # Same for 0-series
            seq_count[i,2] = ((1-(sum(decisions[i])/(1.*min_trials*
                               word_length)))**(word_length))*min_trials
            seq_count[i,3] = sum(sum(decisions[i],0)==0)

        bayes_stat.pred_pos = pred_pos_old
        bayes_stat.report(c,sorn)
        return output_dist        
        
class OutputDistStat(AbstractStat):
    """
    This stat reports the distance between output gains as an indicator
    for whether the decision is based on chance or on attractor dynamics
    """
    def __init__(self):
        self.name = 'OutputDist'
        self.collection = 'gather'
    def report(self,c,sorn):
        # Read stuff in
        letters_for_frac = c.letters_for_frac
        raw_predictions = c.raw_predictions
        inputi_test = c.inputi_test
        pred_pos = c.pred_pos
        if isinstance(sorn.source,TrialSource): # if TrialSource
            source = sorn.source.source
        else:
            source = sorn.source
        N_fracs = len(sorn.c.frac_A)
        
        summed = abs(raw_predictions[:,0])+abs(raw_predictions[:,1])
        summed[summed<1e-10] = 1 # if predicted 0, leave at 0
        raw_predictions[:,0] /= summed
        raw_predictions[:,1] /= summed
        
        output_dist = np.zeros((N_fracs))
        output_std = np.zeros((N_fracs))
        denom = np.zeros((N_fracs))
        for (s_word,s_index) in zip(source.words,source.glob_ind):
            i = ''.join(letters_for_frac).find(s_word[0])
            indices = find(inputi_test==s_index+pred_pos)
            tmp = abs(raw_predictions[indices,0]-
                      raw_predictions[indices,1])
            output_dist[i] += mean(tmp)
            output_std[i]   +=  std(tmp)
            denom[i]          += 1
        output_dist /= denom
        output_std /= denom
        to_return = vstack((output_dist,output_std))
        return to_return

class TrialBayesStat(AbstractStat):
    """
    This stat looks at the interaction of spontaneous activity before
    stimulus onset with the final prediction
    
    index: int
        Word index (global) for which prediction is done
    """
    def __init__(self):
        self.name = 'TrialBayes'
        self.collection = 'gather'
    def report(self,c,sorn):
        # Read stuff in
        STA_window = 50
        pred_pos = c.pred_pos
        classifier_old = c.classifier
        noinput_units = c.noinput_units
        steps_plastic = sorn.c.steps_plastic
        N_train_steps = sorn.c.steps_noplastic_train
        N_inputtrain_steps = steps_plastic + N_train_steps
        N_test_steps = sorn.c.steps_noplastic_test
        # Transpose because this is the way they are in test_bayes.py
        # Use all neurons because we're predicting from spont activity
        Xtest = c.spikes[:,N_inputtrain_steps:].T
        inputi_test = c.inputindex[N_inputtrain_steps:]
        N_exc = shape(Xtest)[1]
        if isinstance(sorn.source,TrialSource): # if TrialSource
                source = sorn.source.source
        else:
            raise NotImplementedError
        
         # select middle word
        index = source.glob_ind[1+(shape(source.glob_ind)[0]-3)//2]
        forward_pred = sorn.c.stats.forward_pred
        start_indices = find(inputi_test==index)
        # * is element-wise AND
        start_indices = start_indices[(start_indices>STA_window) * 
          ((start_indices+pred_pos+forward_pred)<shape(inputi_test)[0])]
        N_samples = shape(start_indices)[0]
        pred_indices = find(inputi_test==(index+pred_pos))
        pred_indices = pred_indices[(pred_indices>=start_indices[0])*
                    ((pred_indices+forward_pred)<shape(inputi_test)[0])]
        assert(N_samples == shape(pred_indices)[0])
        
        if sorn.c.stats.bayes_noinput:
            raw_predictions = Xtest[:,noinput_units].dot(classifier_old)
        else:
            raw_predictions = Xtest.dot(classifier_old)
        predictions = raw_predictions[pred_indices,:]
        
        # Two different baselines
        #~ test_base = ones((shape(Xtest)[0],1))
        test_base = Xtest.copy()
        shuffle(test_base) # without shuffle, identical predictions
        test_base = hstack((test_base,ones((shape(Xtest)[0],1))))
        
        # Add bias term to exclude effects of varability
        N_exc += 1
        Xtest = hstack((Xtest,ones((shape(Xtest)[0],1))))
        
        # Divide into train and test set
        predictions_train = predictions[:N_samples//2]
        predictions_test = predictions[N_samples//2:]
        
        train_A = predictions_train[:,0]>predictions_train[:,1]
        train_B = train_A==False
        train_A = find(train_A==True)
        train_B = find(train_B==True)

        # This case is filtered out during plotting
        if not(shape(train_A)[0]>0 and shape(train_B)[0]>0):
            return np.ones((2,STA_window))*-1
        
        agreement_lstsq = np.zeros(STA_window)     
        agreement_base = np.zeros(STA_window)     
        # This maps 0/1 spikes to -1/1 spikes for later * comparison   
        predtrain_lstsq = (predictions_train[:,0]>\
                                             predictions_train[:,1])*2-1
        predtest_lstsq = (predictions_test[:,0]>\
                                              predictions_test[:,1])*2-1
        # Prediction with spontaneous activity
        for i in range(-STA_window,0):
            classifier_lstsq = lstsq_reg(Xtest[\
                        start_indices[:N_samples//2]+i+forward_pred,:],\
                        predtrain_lstsq,sorn.c.stats.lstsq_mue)
            predictions_lstsq = (Xtest[start_indices[N_samples//2:]+i\
                                 +forward_pred,:]).dot(classifier_lstsq)
                                     # this is where the -1/1 comes in
            agreement_lstsq[i] = sum((predictions_lstsq*predtest_lstsq)\
                                                   >0)/(1.*N_samples//2)
        
        # Baseline prediction (loop is unnecessary and for similarity)
        for i in range(-STA_window,0):
            classifier_base = lstsq_reg(test_base[\
                        start_indices[:N_samples//2]+i+forward_pred,:],\
                        predtrain_lstsq,sorn.c.stats.lstsq_mue)
            predictions_base = (test_base[start_indices[N_samples//2:]+i\
                                 +forward_pred,:]).dot(classifier_base)
            agreement_base[i] = sum((predictions_base*predtest_lstsq)\
                                                   >0)/(1.*N_samples//2)

        # STA - not used
        trials = np.zeros((N_samples,STA_window,N_exc))
        for i in range(N_samples):
            trials[i,:,:] = Xtest[start_indices[i]-STA_window\
                          +forward_pred:start_indices[i]+forward_pred,:]
                          
        STA_A = mean(trials[train_A,:,:],0)
        STA_B = mean(trials[train_B,:,:],0)

        N_test = N_samples-N_samples//2
        overlap_A = np.zeros((N_test,STA_window,N_exc))
        overlap_B = np.zeros((N_test,STA_window,N_exc))
        for i in range(N_samples//2,N_samples):
            overlap_A[i-N_samples//2] = trials[i]*STA_A
            overlap_B[i-N_samples//2] = trials[i]*STA_B
        
        agreement = np.zeros(STA_window)
        pred_gain_A = predictions_test[:,0]>predictions_test[:,1]
        for i in range(STA_window):
            pred_STA_A = sum(overlap_A[:,i,:],1)>sum(overlap_B[:,i,:],1)
            agreement[i] = sum(pred_gain_A == pred_STA_A)
        agreement /= float(shape(pred_gain_A)[0])
        
        return vstack((agreement_base, agreement_lstsq))
    
class SpontBayesStat(AbstractStat):
    def __init__(self):
        self.name = 'SpontBayes'
        self.collection = 'gather'
    def report(self,c,sorn):
        # Read stuff in
        pred_pos = c.pred_pos
        inputi_test = c.inputi_test
        raw_predictions = c.raw_predictions
        Xtest = c.Xtest
        
        # Filter out empty states
        if isinstance(sorn.source,TrialSource): # if TrialSource
            source = sorn.source.source
        else:
            source = sorn.source
        Xtest = Xtest[inputi_test != -1,:]
        inputi_test = inputi_test[inputi_test != -1]
        
        
        letters_for_frac = c.letters_for_frac
        # Results will first be saved in dict for simplicity and later 
        # subsampled to an array
        cue_act = {}
        pred_gain = {}
        minlen = inf
        for (s_word,s_index) in zip(source.words,source.glob_ind):
            i = ''.join(letters_for_frac).find(s_word[0])
            # Indices that point to the presentation of the cue relative
            # to the readout
            cue_indices = find(inputi_test==s_index)
            pred_indices = cue_indices+pred_pos
            pred_indices = pred_indices[pred_indices
                                        <shape(inputi_test)[0]]
            # Get x-states at cue_indices and figure out the number of 
            # active input units for A and B
            tmp_cue = Xtest[cue_indices]
            tmp_cue = vstack((
                        sum(tmp_cue[:,1==sorn.W_eu.W[:,
                                      source.lookup['A']]],1),
                        sum(tmp_cue[:,1==sorn.W_eu.W[:,
                                      source.lookup['B']]],1))).T
            tmp_gain = raw_predictions[pred_indices,:]
            if cue_act.has_key(i):
                cue_act[i] = np.append(cue_act[i],tmp_cue,axis=0)
                pred_gain[i] = np.append(pred_gain[i],tmp_gain,axis=0)
            else:
                cue_act[i] = tmp_cue
                pred_gain[i] = tmp_gain
            if shape(cue_act[i])[0]<minlen:
                minlen = shape(cue_act[i])[0]
        
        # TODO super ugly - try to make prettier
        minlen = 18 # hack for cluster - otherwise variable minlen 
        # subsample to make suitable for array
        n_conditions = max(cue_act.keys())+1
        to_return = np.zeros((n_conditions,minlen,4))
        for i in range(n_conditions):
            to_return[i,:,:2] = cue_act[i][-minlen:]
            to_return[i,:,2:] = pred_gain[i][-minlen:]
        return to_return
        
class EvokedPredStat(AbstractStat):
    """
    This stat predicts evoked activity from spontaneous activity
        
    traintimes is an interval of training data
    testtimes is an interval of testing data
    """
    def __init__(self,traintimes,testtimes,traintest):
        self.name = 'EvokedPred'
        self.collection = 'gather'
        self.traintimes = traintimes
        self.testtimes = testtimes
        self.traintest = traintest
    def report(self,c,sorn):
        # Read data
        traintimes = self.traintimes
        testtimes  = self.testtimes
        Xtrain = c.spikes[:,traintimes[0]:traintimes[1]].T
        Xtest = c.spikes[:,testtimes[0]:testtimes[1]].T
        inputi_train = c.inputindex[traintimes[0]:traintimes[1]]
        inputi_test = c.inputindex[testtimes[0]:testtimes[1]]
        
        # Determine word length
        source = load_source("source_%s"%self.traintest,sorn.c)
        N_words = len(source.words)
        max_word_length = int(max([len(x) for x in source.words]))
        max_spont_length = int(sorn.c['wait_min_%s'%self.traintest]
                              +sorn.c['wait_var_%s'%self.traintest])
        
        pred_window = max_word_length + max_spont_length+max_word_length
        
        correlations = zeros((N_words,pred_window,2))
        import scipy.stats as stats
        
        # Convert 0/1 spike trains to -1/1 spike trains if needed
        if sorn.c.stats.match:
            Xtrain *= 2
            Xtrain -= 1
            Xtest *= 2
            Xtest -= 1
        
        word_length = 0
        for (w,word) in enumerate(source.words):
            word_starts_train = find(inputi_train==(word_length))
            word_starts_train = word_starts_train[(word_starts_train>0)\
                    *(word_starts_train<(shape(Xtrain)[0]-pred_window))]
            word_starts_test  = find(inputi_test==(word_length))
            word_starts_test  = word_starts_test[word_starts_test<\
                                          (shape(Xtest)[0]-pred_window)]
                                          
            bias_train = ones((shape(word_starts_train)[0],1))
            bias_test = ones((shape(word_starts_test)[0],1))
            
            base_train = Xtrain[word_starts_train-1,:].copy() 
            base_test = Xtest[word_starts_test-1,:].copy()
            shuffle(base_train)
            shuffle(base_test)
            base_train = hstack((bias_train,base_train))
            base_test = hstack((bias_test,base_test))

            sp_train = hstack((bias_train,Xtrain[word_starts_train-1,:]))
            sp_test  = hstack((bias_test,Xtest[word_starts_test-1,:]))
            #~ sp_train = bias_train <-- this is a STA!
            #~ sp_test = bias_test
            for t in range(pred_window):
                # First do a least-squares fit
                Xt_train = Xtrain[word_starts_train+t,:]
                Xt_test  = Xtest[word_starts_test+t,:]
                
                # regularize with mue to avoid problems when #samples <
                # #neurons
                classifier = lstsq_reg(sp_train,Xt_train,
                                                 sorn.c.stats.lstsq_mue)
                classifier_base = lstsq_reg(base_train,Xt_train,
                                                 sorn.c.stats.lstsq_mue)
                Xt_pred = sp_test.dot(classifier)
                base_pred = base_test.dot(classifier)
                # Baseline = STA
                #~ base = mean(Xt_train,0)
                #~ base_pred = array([base,]*shape(Xt_test)[0])
                # Don't use this because the paper uses correlation
                # Don't use this because of lower bound for zeros
                # instead of pearsonr - lower bound = 1-h.ip
                # -> spont pred always better
                def match(x,y): 
                    assert(shape(x) == shape(y))
                    x = x>0
                    y = y>0
                    return sum(x==y)/(1.0*shape(x)[0])
                if not sorn.c.stats.match:
                    correlations[w,t,0] = stats.pearsonr(
                                 Xt_pred.flatten(),Xt_test.flatten())[0]
                    correlations[w,t,1] = stats.pearsonr(
                               base_pred.flatten(),Xt_test.flatten())[0]
                else:
                    correlations[w,t,0] = match(Xt_pred.flatten(),
                                                      Xt_test.flatten())
                    correlations[w,t,1] = match(base_pred.flatten(),
                                                      Xt_test.flatten())
            word_length += len(word)
        # Correlations are sorted like the words:
        # A B C D E ... B = 0*A C = 0.1*A, D=0.2*A ...
        return correlations
        
        
class SpikesStat(AbstractStat):
    def __init__(self,inhibitory = False):
        if inhibitory:
            self.name = 'SpikesInh'
            self.sattr = 'spikes_inh'
        else:
            self.name = 'Spikes'
            self.sattr = 'spikes'
        self.collection = 'gather'
        self.inh = inhibitory
    def clear(self,c,sorn):
        if self.inh:
            self.neurons = sorn.c.N_i
        else:
            self.neurons = sorn.c.N_e
        if sorn.c.stats.has_key('only_last'):
            steps = sorn.c.stats.only_last+sorn.c.stats.only_last
            c[self.sattr] = zeros((self.neurons,steps))
        else:
            c[self.sattr] = zeros((self.neurons,sorn.c.N_steps))
        self.step = 0
    def add(self,c,sorn):
        if self.inh:
            spikes = sorn.y
        else:
            spikes = sorn.x
        if sorn.c.stats.has_key('only_last'):
            new_step = self.step - (sorn.c.N_steps\
                                    -sorn.c.stats.only_last)
            if new_step >= 0:
                c[self.sattr][:,new_step+sorn.c.stats.only_last] \
                    = spikes
            elif self.step % (sorn.c.N_steps\
                              //sorn.c.stats.only_last) == 0:
                c[self.sattr][:,self.step//(sorn.c.N_steps\
                //sorn.c.stats.only_last)] = spikes
        else:
            c[self.sattr][:,self.step] = spikes
        self.step += 1
    def report(self,c,sorn):
        if sorn.c.stats.save_spikes:
            return c[self.sattr]
        else:
            return zeros(0)
            
class CondProbStat(AbstractStat):
    def __init__(self):
        self.name='CondProb'
        self.collection='gather'
    def clear(self,c,sorn):
        pass
    def add(self,c,sorn):
        pass
    def report(self,c,sorn):
        # return a marix with M_ij = frequency of a spike in i following
        # a spike in j
        # Look at test instead of training to get more diverse data
        steps = sorn.c.steps_noplastic_test
        spikes = c.spikes[:,-steps:]
        N = shape(spikes)[0] # number of neurons
        condspikes = np.zeros((N,N))
        for t in xrange(1,steps):
            condspikes[spikes[:,t]==1,:] += spikes[:,t-1]
        spike_sum = sum(spikes,1)
        for i in xrange(N):
            condspikes[i,:] /= spike_sum
        return condspikes
        
class BalancedStat(AbstractStat):
    """
    This stat records the excitatory and inhibitory input and thresholds
    to determine how balanced the network operates
    """
    def __init__(self):
        self.name='Balanced'
        self.collection='gather'
    def clear(self,c,sorn):
        c.balanced = zeros((sorn.c.N_e*3,sorn.c.N_steps))
        self.step = 0
        self.N_e = sorn.c.N_e
    def add(self,c,sorn):
        c.balanced[:self.N_e,self.step] = sorn.W_ee*sorn.x
        c.balanced[self.N_e:2*self.N_e,self.step] = sorn.W_ei*sorn.y
        c.balanced[2*self.N_e:,self.step] = sorn.T_e
        self.step += 1
    def report(self,c,sorn):
        return c.balanced
        
class RateStat(AbstractStat):
    """
    This stat returns a matrix of firing rates of each presynaptic
    neuron
    """
    def __init__(self):
        self.name = 'Rate'
        self.collection='gather'
    def clear(self,c,sorn):
        pass
    def add(self,c,sorn):
        pass
    def report(self,c,sorn):
        # same interval as for condprob
        steps = sorn.c.steps_noplastic_test
        spikes = c.spikes[:,-steps:]
        N = shape(spikes)[0] # number of neurons
        rates = mean(spikes,1)
        return array([rates,]*N)
        
class InputStat(AbstractStat):
    def __init__(self):
        self.name = 'Input'
        self.collection = 'gather'
    def clear(self,c,sorn):
        c.inputs = zeros((sorn.c.N_e,sorn.c.N_steps))
        self.step = 0
    def add(self,c,sorn):
        c.inputs[:,self.step] = sorn.W_eu*sorn.u
        self.step += 1
    def report(self,c,sorn):
        return c.inputs
        

class FullEndWeightStat(AbstractStat):
    def __init__(self):
        self.name = 'FullEndWeight'
        self.collection = 'gather'
    def clear(self,c,sorn):
        pass
    def add(self,c,sorn):
        pass
    def report(self,c,sorn):
        tmp1 = np.vstack((sorn.W_ee.get_synapses(),\
                          sorn.W_ie.get_synapses()))
        tmp2 = np.vstack((sorn.W_ei.get_synapses(),\
                          np.zeros((sorn.c.N_i,sorn.c.N_i))))
        return np.array(hstack((tmp1,tmp2)))
        

class EndWeightStat(AbstractStat):
    def __init__(self):
        self.name = 'endweight'
        self.collection = 'gather'
    def clear(self,c,sorn):
        pass
    def add(self,c,sorn):
        pass
    def report(self,c,sorn):
        if sorn.c.W_ee.use_sparse:
            return np.array(sorn.W_ee.W.todense())
        else:
            return sorn.W_ee.W*(sorn.W_ee.M==1)


class ISIsStat(AbstractStat):
    def __init__(self,interval=[]):
        self.name = 'ISIs'
        self.collection = 'gather'
        self.interval = interval
    def clear(self,c,sorn):
        self.mask = sum(sorn.W_eu.get_synapses(),1)==0
        self.N_noinput = sum(self.mask)
        self.ISIs = zeros((self.N_noinput,100))
        self.isis = zeros(self.N_noinput)
        self.step = 0
        if self.interval == []:
            self.interval = [0,sorn.c.N_steps]
    def add(self,c,sorn):
        if ((self.step > self.interval[0] and 
             self.step < self.interval[1]) and 
             ((not sorn.c.stats.has_key('only_last')) \
                or (self.step > sorn.c.stats.only_last))):
            spikes = sorn.x[self.mask]
            self.isis[spikes==0] += 1
            isis_tmp = self.isis[spikes==1]
            isis_tmp = isis_tmp[isis_tmp<100]
            tmp = zip(where(spikes==1)[0],isis_tmp.astype(int))
            for pair in tmp:
                self.ISIs[pair] += 1
            self.isis[spikes==1] = 0
        self.step += 1
    def report(self,c,sorn):
        return self.ISIs

class SynapseFractionStat(AbstractStat):
    def __init__(self):
        self.name = 'SynapseFraction'
        self.collection = 'reduce'
    def report(self,c,sorn):
        if sorn.c.W_ee.use_sparse:
            return array(sum((sorn.W_ee.W.data>0)+0.0)\
                            /(sorn.c.N_e*sorn.c.N_e))
        else:
            return array(sum(sorn.W_ee.M)/(sorn.c.N_e*sorn.c.N_e))

class ConnectionFractionStat(AbstractStat):
    def __init__(self):
        self.name = 'ConnectionFraction'
        self.collection = 'gather'
    def clear(self,c,sorn):
        self.step = 0
        if sorn.c.stats.has_key('only_last'):
            self.cf = zeros(sorn.c.stats.only_last\
                            +sorn.c.stats.only_last)
        else:
            self.cf = zeros(sorn.c.N_steps)
    def add(self,c,sorn):
        if sorn.c.stats.has_key('only_last'):
            new_step = self.step \
                        - (sorn.c.N_steps-sorn.c.stats.only_last)
            if new_step >= 0:
                if sorn.c.W_ee.use_sparse:
                    self.cf[new_step+sorn.c.stats.only_last] = sum(\
                        (sorn.W_ee.W.data>0)+0)/(sorn.c.N_e*sorn.c.N_e)
                else:
                    self.cf[new_step+sorn.c.stats.only_last] = sum(\
                                    sorn.W_ee.M)/(sorn.c.N_e*sorn.c.N_e)
            elif self.step%(sorn.c.N_steps\
                    //sorn.c.stats.only_last) == 0:
                if sorn.c.W_ee.use_sparse:
                    self.cf[self.step//(sorn.c.N_steps\
                        //sorn.c.stats.only_last)] = sum(\
                        (sorn.W_ee.W.data>0)+0)/(sorn.c.N_e*sorn.c.N_e)
                else:
                    self.cf[self.step//(sorn.c.N_steps\
                        //sorn.c.stats.only_last)] = sum(\
                        sorn.W_ee.M)/(sorn.c.N_e*sorn.c.N_e)
        else:
            if sorn.c.W_ee.use_sparse:
                self.cf[self.step] = sum((sorn.W_ee.W.data>0)+0)\
                                        /(sorn.c.N_e*sorn.c.N_e)
            else:
                self.cf[self.step] = sum(sorn.W_ee.M)\
                                        /(sorn.c.N_e*sorn.c.N_e)
        self.step += 1
    def report(self,c,sorn):
        return self.cf

class WeightLifetimeStat(AbstractStat):
    def __init__(self):
        self.name = 'WeightLifetime'
        self.collection = 'gather'
    def clear(self,c,sorn):
        if sorn.c.W_ee.use_sparse:
            self.last_M_ee = np.array(sorn.W_ee.W.todense())>0
        else:
            self.last_M_ee = sorn.W_ee.M.copy()
        self.lifetimes = zeros((sorn.c.N_e,sorn.c.N_e))
        self.diedat = np.zeros((1,0))
    def add(self,c,sorn):
        if sorn.c.W_ee.use_sparse:
            new_M_ee = np.array(sorn.W_ee.W.todense())>0
        else:
            new_M_ee = sorn.W_ee.M
        self.diedat = append(self.diedat, \
                      self.lifetimes[(new_M_ee+0-self.last_M_ee+0)==-1])
        # remove dead synapses
        self.lifetimes *= new_M_ee+0
        #increase lifetime of existing ones
        self.lifetimes += (self.lifetimes>0)+0
        #add new ones
        self.lifetimes += ((new_M_ee+0-self.last_M_ee+0)==1)+0

        self.last_M_ee = new_M_ee.copy()
    def report(self,c,sorn):
        padding = (-1)*np.ones(2*sorn.c.N_steps\
                        +shape(self.last_M_ee)[0]**2-self.diedat.size)
        return np.append(self.diedat,padding)

class WeightChangeStat(AbstractStat):
    def __init__(self):
        self.name = 'WeightChange'
        self.collection = 'gather'
    def clear(self,c,sorn):
        self.step = 0
        self.start = 2999
        self.end = 5999
        self.save_W_ee = []
        self.abschange = []
        self.relchange = []
        self.weights = []
    def add(self,c,sorn):
        if(self.step == self.start):
            if sorn.c.W_ee.use_sparse:
                self.save_W_ee = np.array(sorn.W_ee.W.todense())
            else:
                self.save_W_ee = sorn.W_ee.W.copy()
        if(self.step == self.end):
            if sorn.c.W_ee.use_sparse:
                diff = np.array(sorn.W_ee.W.todense())-self.save_W_ee
            else:
                diff = sorn.W_ee.W-self.save_W_ee
            self.weights = self.save_W_ee[diff!=0]
            self.abschange = (diff[diff!=0])
            seterr(divide='ignore')
            # Some weights become 0 and thereby elicit division by 0
            # and try except RuntimeWarning didn't work
            self.relchange = self.abschange/self.weights*100
            seterr(divide='warn')
            # append zeros to always have the same size
            tmp_zeros = np.zeros(shape(self.save_W_ee)[0]**2\
                                 -self.weights.size)
            self.weights = np.append(self.weights,tmp_zeros)
            self.abschange = np.append(self.abschange,tmp_zeros)
            self.relchange = np.append(self.relchange,tmp_zeros)
        self.step += 1
    def report(self,c,sorn):
        stacked = np.vstack((self.weights, self.abschange,\
                            self.relchange))
        return stacked
        
class WeightChangeRumpelStat(AbstractStat):
    def __init__(self):
        self.name = 'WeightChangeRumpel'
        self.collection = 'gather'
    def clear(self,c,sorn):
        self.step = 0
        self.interval = 0
        self.start = 50001
        self.started = False
        self.imaging_interval = 50000
        self.N_intervals = (sorn.c.N_steps-self.start)\
                            //self.imaging_interval+1
        self.save_W_ees = np.zeros((self.N_intervals,sorn.c.N_e,\
                                    sorn.c.N_e))
        self.constant_weights = []
        self.abschange = []
        self.relchange = []
        self.weights = []
    def add(self,c,sorn):
        if(self.step%self.imaging_interval == 0 and self.started):
            self.save_W_ees[self.interval,:,:] \
                    = sorn.W_ee.get_synapses()
            self.constant_weights *= (self.save_W_ees[self.interval,\
                                                      :,:]>0)
            self.interval += 1
        if(self.step == self.start):
            self.save_W_ees[self.interval,:,:] \
                    = sorn.W_ee.get_synapses()
            self.constant_weights \
                    = (self.save_W_ees[self.interval,:,:].copy()>0)
            self.interval = 1
            self.started = True
        self.step += 1
    def report(self,c,sorn):
        # compute diffs and multiply with const
        import pdb
        pdb.set_trace()
        
        diffs = self.save_W_ees[1:,:,:] - self.save_W_ees[:-1,:,:]
        diffs *= self.constant_weights
        
        self.abschange = (diffs[diffs!=0])
        self.weights = self.save_W_ees[:-1,:,:][diffs!=0]
        self.relchange = self.abschange/self.weights*100
        # append zeros to always have the same size
        tmp_zeros = np.zeros((self.N_intervals-1)\
                     *shape(self.save_W_ees)[1]**2-self.weights.size)
        self.weights = np.append(self.weights,tmp_zeros)
        self.abschange = np.append(self.abschange,tmp_zeros)
        self.relchange = np.append(self.relchange,tmp_zeros)
        
        stacked = np.vstack((self.weights, self.abschange,\
                             self.relchange))
        return stacked

class SmallWorldStat(AbstractStat):
    def __init__(self):
        self.name = 'smallworld'
        self.collection = 'gather'
    def clear(self,c,sorn):
        pass
    def add(self,c,sorn):
        pass
    def report(self,c,sorn):
        if sorn.c.stats.rand_networks <= 0:
            return np.array([])
        if sorn.c.W_ee.use_sparse:
            weights = np.array(sorn.W_ee.W.todense())
        else:
            weights = sorn.W_ee.W*(sorn.W_ee.M==1)
        tmp = weights>0.0+0.0
        binary_connections = tmp+0.0
        
        def all_pairs_shortest_path(graph_matrix):
            # adapted Floyd-Warshall Algorithm
            N = shape(graph_matrix)[0]
            distances = graph_matrix.copy()
            #Set missing connections to max length
            distances[distances==0] += N*N 
            for k in range(N):
                for i in range(N):
                    for j in range(N):
                        if i==j:
                            distances[i,j] = 0
                        else:
                            distances[i,j] = min(distances[i,j],
                                                 distances[i,k]
                                                   +distances[k,j])
            return distances

        def characteristic_path_length(graph_matrix):
            N = shape(graph_matrix)[0]
            distances = all_pairs_shortest_path(graph_matrix.T)
            if any(distances == N*N):
                print 'Disconnected elements in char. path len calc.'
            # ignore disconnected elements
            distances[distances==N*N] = 0
            average_length = sum(distances[distances>0]*1.0)\
                            /sum(graph_matrix[distances>0]*1.0)
            return average_length

        def cluster_coefficient(graph_matrix):
            # From Fagiolo, 2007 and Gerhard, 2011
            N = shape(graph_matrix)[0]
            in_degree = sum(graph_matrix,1)
            out_degree = sum(graph_matrix,0)
            k = in_degree+out_degree
            A = graph_matrix
            A_T = A.transpose()
            A_A_T = A + A_T
            A_2 = np.dot(A,A)
            nominator = np.dot(A_A_T,np.dot(A_A_T,A_A_T))
            single_coeff = np.zeros(N)
            for i in range(N):
                single_coeff[i] = nominator[i,i]/(2.0*(k[i]*(k[i]-1)\
                                                 -2.0*(A_2[i,i])))
                if(np.isnan(single_coeff[i])):
                    # if total degree <= 1, the formula divides by 0
                    single_coeff[i] = 0
            return 1.0*sum(single_coeff)/(N*1.0)

        L = characteristic_path_length(binary_connections)
        C = cluster_coefficient(binary_connections)
        # Average over some random networks
        N = shape(binary_connections)[0]
        edge_density = sum(binary_connections)/(1.0*N*N-N)
        num_rand = sorn.c.stats.rand_networks
        L_rand = np.zeros(num_rand)
        C_rand = np.zeros(num_rand)
        delete_diagonal = np.ones((N,N))
        for i in range(N):
            delete_diagonal[i,i] = 0
        for i in range(num_rand):
            sys.stdout.write('\rRand Graph No.%3i of %3i'%(i+1,\
                                                            num_rand))
            sys.stdout.flush()
            tmp = np.random.rand(N,N)<edge_density
            rand_graph = tmp*delete_diagonal
            L_rand[i] = characteristic_path_length(rand_graph)
            C_rand[i] = cluster_coefficient(rand_graph)
        sys.stdout.write('\rAll %i Graphs Done       '%num_rand)
        sys.stdout.flush()
        L_r = sum(L_rand)*1.0/(num_rand*1.0)
        C_r = sum(C_rand)*1.0/(num_rand*1.0)

        gamma = C/C_r
        lam = L/L_r
        S_w = gamma/lam
        return np.array([gamma, lam, S_w])

class ParamTrackerStat(AbstractStat):
    def __init__(self):
        self.name = 'paramtracker'
        self.collection = 'gather'
    def clear(self,c,sorn):
        pass
    def add(self,c,sorn):
        pass
    def report(self,c,sorn):
        tmp = sorn.c
        for item in sorn.c.cluster.vary_param.split('.'):
            tmp = tmp[item]
        return np.array([tmp*1.0])
        
class InputWeightStat(AbstractStat):
    def __init__(self):
        self.name = 'InputWeight'
        self.collection = 'gather'
    def clear(self,c,sorn):
        self.step = 0
        self.weights = np.zeros((sorn.c.N_e,sorn.c.N_u_e,\
                                 sorn.c.stats.only_last*2))
    def add(self,c,sorn):
        if self.step % (sorn.c.N_steps//sorn.c.stats.only_last) == 0:
            self.weights[:,:,self.step//(sorn.c.N_steps\
                  //sorn.c.stats.only_last)] = sorn.W_eu.get_synapses()
        self.step += 1
    def report(self,c,sorn):
        return self.weights

class SVDStat(AbstractStat):
    def __init__(self,nth = 200):
        self.name = 'SVD'
        self.collection = 'gather'
        self.nth = nth
    def clear(self,c,sorn):
        self.step = 0
        # Quick hack - there must be a prettier solution
        if sorn.c.steps_plastic % self.nth == 0:
            add1 = 0
        else:
            add1 = 1
        c.SVD_singulars = np.zeros((sorn.c.steps_plastic//self.nth+add1
                                                           ,sorn.c.N_e))
        c.SVD_U = np.zeros((sorn.c.steps_plastic//self.nth+add1,
                            sorn.c.N_e,sorn.c.N_e))
        c.SVD_V = np.zeros((sorn.c.steps_plastic//self.nth+add1,
                            sorn.c.N_e,sorn.c.N_e))
    def add(self,c,sorn):
        if self.step < sorn.c.steps_plastic and self.step%self.nth == 0:
            # Time intensive!
            synapses = sorn.W_ee.get_synapses()
            U,s,V = linalg.svd(synapses)
            c.SVD_singulars[self.step//self.nth,:] = s
            step = self.step//self.nth
            c.SVD_U[step] = U
            # this returns the real V
            # see http://docs.scipy.org/doc/numpy/reference/generated/numpy.linalg.svd.html
            c.SVD_V[step] = V.T
            
            # Resolve sign ambiguity
            # from http://www.models.life.ku.dk/signflipsvd
            # http://prod.sandia.gov/techlib/access-control.cgi/2007/076422.pdf
            for i in range(sorn.c.N_e):
                tmp = synapses.T.dot(c.SVD_U[step,:,i])
                tmp = np.squeeze(asarray(tmp))
                s_left = sum(sign(tmp)*tmp**2)
                tmp = synapses.T.dot(c.SVD_V[step,:,i])
                tmp = np.squeeze(asarray(tmp))
                s_right = sum(sign(tmp)*tmp**2)
                if s_right*s_left < 0:
                    if s_left < s_right:
                        s_left = -s_left
                    else:
                        s_right = -s_right
                c.SVD_U[step,:,i] *= sign(s_left)
                c.SVD_V[step,:,i] *= sign(s_right)
        self.step += 1
    def report(self,c,sorn):
        #~ figure() # combine same submatrices!
        #~ imshow(c.SVD_U[-1][:,0].dot(c.SVD_V[-1][:,0].T)\
        #~ *c.SVD_singulars[-1,0], interpolation='none')
        return c.SVD_singulars
        
class SVDStat_U(AbstractStat):
    def __init__(self):
        self.name = 'SVD_U'
        self.collection = 'gather'
    def report(self,c,sorn):
        rec_steps = shape(c.SVD_U)[0]
        similar_input = zeros((rec_steps,sorn.c.N_e))
        N_indices = max(c.norm_last_input_index)+1
        indices = [where(c.norm_last_input_index==i)[0] for i in 
                                                  range(int(N_indices))]
        for s in xrange(rec_steps):
            for i in xrange(sorn.c.N_e):
                # U transforms back to "spike space"
                # Check for best similarities
                # Convolution works best:
                #~ overlaps = c.norm_last_input_spikes.T.dot(
                                                    #~ c.SVD_U[s,:,i])
                #~ index_overlap = np.zeros(N_indices)
                #~ for j in range(int(N_indices)):
                    #~ index_overlap[j] = mean(overlaps[indices[j]])
                #~ similar_input[s,i] = argmax(index_overlap)
                # No big difference to this, but probably more robust
                max_overlap = argmax(c.norm_last_input_spikes.T.dot(
                                                        c.SVD_U[s,:,i]))
                similar_input[s,i] = c.norm_last_input_index[
                                                            max_overlap]
        c.SVD_U_sim = similar_input # for debugging
        return similar_input
        
class SVDStat_V(AbstractStat):
    def __init__(self):
        self.name = 'SVD_V'
        self.collection = 'gather'
    def report(self,c,sorn):
        rec_steps = shape(c.SVD_V)[0]
        similar_input = zeros((rec_steps,sorn.c.N_e))
        N_indices = max(c.norm_last_input_index)+1
        indices = [where(c.norm_last_input_index==i)[0] for i in 
                                                  range(int(N_indices))]
        for s in xrange(rec_steps):
            for i in xrange(sorn.c.N_e):                
                # V transforms input by taking product
                # Do same here and look which spike vector works best
                #~ overlaps = c.norm_last_input_spikes.T.dot(
                                                    #~ c.SVD_V[s,:,i])
                #~ index_overlap = np.zeros(N_indices)
                #~ for j in range(int(N_indices)):
                    #~ index_overlap[j] = mean(overlaps[indices[j]])
                #~ similar_input[s,i] = argmax(index_overlap)
                # No big difference to this, but probably more robust
                max_overlap = argmax(c.norm_last_input_spikes.T.dot(
                              c.SVD_V[s,:,i])) # euclidean norm w/o sqrt
                similar_input[s,i] = c.norm_last_input_index[
                                                            max_overlap]
        '''
        # For testing purposes command line
        !i = 30
        !similar_input[:,i]
        !c.SVD_U_sim[:,i]
        !figure()
        !plot(c.SVD_V[-1,:,i])
        !max_overlap = argmax(c.norm_last_input_spikes.T.dot(c.SVD_V[s,:,i]))
        !plot(c.norm_last_input_spikes[:,max_overlap])
        !figure()
        !plot(c.SVD_U[-1,:,i])
        !max_overlap = argmax(c.norm_last_input_spikes.T.dot(c.SVD_U[s,:,i]))
        !plot(c.norm_last_input_spikes[:,max_overlap])
        !show()
        '''
        return similar_input
        
class MeanActivityStat(AbstractStat):
    """
    This stat returns the mean activity for each inputindex
    """
    def __init__(self,start,stop,N_indices,LFP=False):
        self._start = start
        self._stop = stop
        self._N_indices = N_indices
        self.name = 'meanactivity'
        self.collection = 'gather'
        self.LFP = LFP
        self.tmp = -1
    def clear(self,c,sorn):
        self.means = zeros(self._N_indices)
        self.counter = zeros(self._N_indices)
        self.step = 0
        self.index = None
    def add(self,c,sorn):
        if self.step > self._start and self.step < self._stop\
            and self.step>0:
            # for proper assignment, blank(-1)->0, 0->1...
            self.index = sorn.source.global_index()+1
            if self.index is not None:
                if self.tmp >= 0:
                    self.counter[self.index] += 1.
                if self.LFP:
                    # save input at current step, but can only compute
                    # input for next step!
                    if self.tmp >= 0:
                        self.means[self.index] += self.tmp+sum(sorn.W_eu
                                                                *sorn.u)
                    self.tmp = sum(sorn.W_ee*sorn.x)
                else:
                    if self.tmp >= 0:
                        self.means[self.index] += sum(sorn.x)
                    self.tmp = 0 # dummy value never used
            #~ # +1 due to -1 for blank trials
            #~ self.index = sorn.source.global_index()+1
        self.step += 1
    def report(self,c,sorn):
        return self.means/self.counter
        
class MeanPatternStat(AbstractStat):
    """
    This stat returns the mean activity for each inputindex
    """
    def __init__(self,start,stop,N_indices):
        self._start = start
        self._stop = stop
        self._N_indices = N_indices
        self.name = 'meanpattern'
        self.collection = 'gather'
    def clear(self,c,sorn):
        self.means = zeros((self._N_indices,sorn.c.N_e))
        self.counter = zeros(self._N_indices)
        self.step = 0
        self.index = None
    def add(self,c,sorn):
        if self.step > self._start and self.step < self._stop\
            and self.step>0:
            # for proper assignment, blank(-1)->0, 0->1...
            self.index = sorn.source.global_index()+1
            if self.index is not None:
                self.counter[self.index] += 1.
                self.means[self.index] += sorn.x
        self.step += 1
    def report(self,c,sorn):
        return self.means/self.counter[:,None]
        
class PatternProbabilityStat(AbstractStat):
    """
    This stat estimates the probability distribution of patterns
    for different time intervals
    
    Intervals: List of 2-entry lists 
        [[start1,stop1],...,[startn,stopn]]
    zero_correction: Bool
        Correct estimates by adding one observation to each pattern
    subset: 1-D array
        List of neuron indices that create the pattern
    """
    def __init__(self,intervals,subset,zero_correction=True):
        self.N_intervals = len(intervals)
        self.intervals = intervals
        self.zero_correction = zero_correction
        self.N_nodes = len(subset)
        self.subset = subset
        self.name = 'patternprobability'
        self.collection = 'gather'
        self.conversion_array = [2**x for x in range(self.N_nodes)][::-1]
        def convert(x):
            return np.dot(x,self.conversion_array)
        self.convert = convert
    def clear(self,c,sorn):
        self.patterns = zeros((self.N_intervals,2**self.N_nodes))
        self.step = 0
    def add(self,c,sorn):
        for (i,(start,stop)) in enumerate(self.intervals):
            if self.step > start and self.step < stop:
                # Convert spiking pattern to integer by taking the
                # pattern as a binary number
                self.patterns[i,self.convert(sorn.x[self.subset])] += 1
        self.step += 1
    def report(self,c,sorn):
        if self.zero_correction:
            self.patterns += 1
        # Normalize to probabilities
        self.patterns /= self.patterns.sum(1)[:,None]
        return self.patterns
        
class WeeFailureStat(AbstractStat):
    def __init__(self):
        self.name = 'weefail'
        self.collection = 'gather'
    def clear(self,c,sorn):
        c.weefail = zeros(sorn.c.N_steps)
        self.step = 0
    def add(self,c,sorn):
        if sorn.c.W_ee.use_sparse:
            N_weights = sorn.W_ee.W.data.shape[0]
            N_fail = N_weights-sum(sorn.W_ee.mask)
        else:
            N_weights = sum(sorn.W_ee.get_synapses()>0)
            N_fail = N_weights-sum(sorn.W_ee.masked>0)
        c.weefail[self.step] = N_fail/N_weights
        self.step += 1
    def report(self,c,sorn):
        return c.weefail
        
class WeeFailureFuncStat(AbstractStat):
    def __init__(self):
        self.name = 'weefailfunc'
        self.collection = 'gather'
    def clear(self,c,sorn):
        self.x = np.linspace(0,1,1000)
        self.y = sorn.W_ee.fail_f(self.x)
    def add(self,c,sorn):
        pass
    def report(self,c,sorn):
        return np.array([self.x,self.y])

# From Philip
class XClassifierStat(AbstractStat):
    def __init__(self,steps=None, classify_x=True, \
                 classify_r=False,detailed=False,**args):
        '''Steps is a list with the step sizes over which to predict.
        e.g.
         - a step of +1 means predict the next state
         - a step of  0 means identify the current state
         - a step of -1 means identify the previous state
         '''
        if steps is None:
            steps = [0]
        self.steps = steps
        self.classify_x = classify_x
        self.classify_r = classify_r
        self.detailed = detailed

    @property
    def name(self):
        ans = []
        if self.classify_x:
            ans.append('xclassifier')
        if self.classify_r:
            ans.append('rclassifier')
        return ans

    def build_classifier(self,inp,out,offset):
        # Use the input to build a classifier of the output with an
        # offset
        N = inp.shape[0]
        inp_aug = hstack([inp, ones((N,1))])
        (ib,ie) = (max(-offset,0),min(N-offset,N))
        (ob,oe) = (max(+offset,0),min(N+offset,N))
        try:
            ans = linalg.lstsq(inp_aug[ib:ie,:],out[ob:oe,:])[0]
        except LinAlgError:
            ans = zeros( (inp.shape[1]+1,out.shape[1]) )
        return ans

    def use_classifier(self,inp,classifier,offset,correct):
        N = inp.shape[0]
        L = classifier.shape[1]
        inp_aug = hstack([inp, ones((N,1))])
        (ib,ie) = (max(-offset,0),min(N-offset,N))
        (ob,oe) = (max(+offset,0),min(N+offset,N))
        ind = argmax(inp_aug[ib:ie,:].dot(classifier),1)
        actual = argmax(correct,1)[ob:oe]
        num = zeros(L)
        den = zeros(L)
        for l in range(L):
            l_ind = actual==l
            num[l] = sum(actual[l_ind]==ind[l_ind])
            den[l] = sum(l_ind)
        return (num,den)

    def report(self,_,sorn):
        c = sorn.c
        #Disable plasticity when measuring network
        sorn.update = False 
        #Don't track statistics when measuring either
        self.parent.disable = True 

        #Build classifiers
        Nr = c.test_num_train
        Nt = c.test_num_test

        #~ (Xr,Rr,Ur) = sorn.simulation(Nr)
        dic = sorn.simulation(Nr,['X','R_x','U'])
        Xr = dic['X']
        Rr = dic['R_x']
        Ur = dic['U']

        #~ (Xt,Rt,Ut) = sorn.simulation(Nt)
        dic = sorn.simulation(Nt,['X','R_x','U'])
        Xt = dic['X']
        Rt = dic['R_x']
        Ut = dic['U']

        L = Ur.shape[1]
        Rr = (Rr >= 0.0)+0
        Rt = (Rt >= 0.0)+0
        r = []
        x = []
        detail_r=[]
        detail_x=[]
        for step in self.steps:
            if self.classify_x:
                classifier = self.build_classifier(Xr,Ur,step)
                (num,den) = self.use_classifier(Xt,classifier,step,Ut)
                ans = sum(num)/sum(den)
                x.append(ans)
                if self.detailed:
                    detail_x.append(num/(den+1e-20))
            if self.classify_r:
                classifier = self.build_classifier(Rr,Ur,step)
                (num,den) = self.use_classifier(Rt,classifier,step,Ut)
                ans = sum(num)/sum(den)
                r.append(ans)
                if self.detailed:
                    detail_r.append(num/(den+1e-20))

        ans = []
        if self.classify_x:
            ans.append( ('xclassifier', 'reduce', array(x)) )
            if self.detailed:
                ans.append( ('x_detail_classifier%d'%L,'reduce',\
                            array(detail_x)) )

        if self.classify_r:
            ans.append( ('rclassifier', 'reduce', array(r)) )
            if self.detailed:
                ans.append( ('r_detail_classifier%d'%L,'reduce',\
                            array(detail_r)) )

        sorn.update = True
        self.parent.disable = False

        return ans

# From Philip
class XTotalsStat(AbstractStat):
    def __init__(self):
        self.name = 'x_tot'
        self.collection = 'gather'
    def clear(self,c,obj):
        N = obj.c.N_e
        c.x_tot = zeros(N)
    def add(self,c,obj):
        c.x_tot += obj.x
    def report(self,c,obj):
        return c.x_tot

# From Philip
class YTotalsStat(AbstractStat):
    def __init__(self):
        self.name = 'y_tot'
        self.collection = 'gather'
    def clear(self,c,obj):
        N = obj.c.N_i
        c.y_tot = zeros(N)
    def add(self,c,obj):
        c.y_tot += obj.y
    def report(self,c,obj):
        return c.y_tot

# From Philip
class SynapticDistributionStat(AbstractStat):
    def __init__(self,collection='gatherv'):
        self.name = 'synaptic_strength'
        self.collection = collection
    def report(self,_,sorn):
        W = sorn.W_ee.T
        Mask = sorn.M_ee.T
        # This code might be a little fragile but fast 
        # (note transposes rely on memory laid out in particular order)
        #~ N = sorn.c.N_e  
        #~ M = sorn.c.lamb
        #This relies on a fixed # of non-zero synapses per neuron
        #~ ans = (W[Mask]).reshape(N,M).T.copy() 
        ans = W[Mask]
        return ans

# From Philip
class SuccessiveStat(AbstractStat):
    def __init__(self):
        self.name = 'successive'
        self.collection = 'reduce'
    def clear(self,c,sorn):
        N = sorn.c.N_e
        c.successive = zeros( (N+1,N+1) )
        c.successive_prev = sum(sorn.x)
    def add(self, c, sorn):
        curr = sum(sorn.x)
        c.successive[c.successive_prev,curr] += 1.0
        c.successive_prev = curr
    def report(self,c,sorn):
        return c.successive
        
# From Philip
class RClassifierStat(AbstractStat):
    def __init__(self,select=None):
        if select is None:
            select = [True,True,True]
        self.name = 'classifier'
        self.collection = 'reduce'
        self.select = select
    def report(self,_,sorn):
        c = sorn.c
        sorn.update = False
        self.parent.disable = True

        #Build classifiers
        N = c.test_num_train
        #~ (X,R,U) = sorn.simulation(N)
        dic = sorn.simulation(N,['X','R_x','U'])
        X = dic['X']
        R = dic['R_x']
        U = dic['U']

        R = hstack([R>=0,ones((N,1))])

        if self.select[0]:
            classifier0 = linalg.lstsq(R,U)[0]
        if self.select[1]:
            classifier1 = dot(linalg.pinv(R),U)
        if self.select[2]:
            X_aug = hstack([X,   ones((N,1))])
            classifier2 = linalg.lstsq(X_aug[:-1,:],U[1:,:])[0]

        #Now test classifiers
        N = c.test_num_test
        #~ (X,R,U) = sorn.simulation(N)
        dic = sorn.simulation(N,['X','R_x','U'])
        X = dic['X']
        R = dic['R_x']
        U = dic['U']

        R = hstack([R>=0,ones((N,1))])

        if self.select[0]:
            ind0 = argmax(dot(R,classifier0),1)
        if self.select[1]:
            ind1 = argmax(dot(R,classifier1),1)
        if self.select[2]:
            X_aug = hstack([X,   ones((N,1))])
            ind2 = argmax(dot(X_aug[:-1,:],classifier2),1)
        actual = argmax(U,1)

        ans = []
        if self.select[0]:
            ans.append(mean(actual==ind0))
        if self.select[1]:
            ans.append(mean(actual==ind1))
        if self.select[2]:
            ans.append(mean(actual[1:]==ind2))

        sorn.update = True
        self.parent.disable = False

        return array(ans)

class WeightHistoryStat(HistoryStat):
    def add(self,c,obj):
        if not (c.history[self.counter] % self.record_every_nth):
            c.history[self.name].append(np.copy(
                                  _getvar(obj,self.var).get_synapses()))
        c.history[self.counter] += 1
