from __future__ import division
from pylab import *
from scipy.optimize import curve_fit
from scipy import stats
import tables

import sys
sys.path.insert(0,"../")
import utils
utils.backup(__file__)

from scipy.io import savemat
import datetime
from utils.pca import pca
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats.stats import pearsonr

import cPickle as pickle
import gzip
from common.sources import TrialSource
import os
import platform
import matplotlib as mpl

from utils.plotting import pretty_mpl_defaults

# Parameters
plot_MDS = False # Takes long
normalize_PCA = False
use_matlab = False # for FF control and matlab-mds (requires mlabwrap) 
plot_spikes = True
ftype = 'pdf' # eps does not support transparency
pca_animation = False

# Data to plot
path = r'/home/chartmann/Desktop/Meeting Plots/2015-12-08_pcaanimations/nolearning_2015-12-08_11-23-31/common'
datafile = 'result.h5'

def parallel_stats(W_ee_h,W_ee2_h):
        diff = abs(W_ee_h-W_ee2_h)
        meandiff = zeros(shape(diff)[0])
        medcv = zeros(shape(diff)[0])
        for i in range(shape(diff)[0]):
            nonzero = (W_ee_h[i]!=0) * (W_ee2_h[i]!=0)
            meandiff[i] = (diff[i][nonzero].mean())# /
                           #W_ee_h[i][W_ee_h[i]!=0].mean())
            stds = std([W_ee_h[i][nonzero],W_ee2_h[i][nonzero]],0)
            means = mean([W_ee_h[i][nonzero],W_ee2_h[i][nonzero]],0)
            medcv[i] = np.median(stds/means)
        # return median development and final means/stds
        return meandiff, medcv, means, stds

def plot_results(result_path,result):
    pretty_mpl_defaults()
    h5 = tables.openFile(os.path.join(result_path,result),'r')
    data = h5.root 
    pickle_dir = data.c.logfilepath[0]
    if not os.path.isdir(pickle_dir):
        pickle_dir = result_path
    plots_path = os.path.join('..','plots')
    if not os.path.exists(plots_path):
        os.mkdir(plots_path)
    os.chdir(plots_path)
    
    ### Plot weight development
    if data.__contains__('W_ee_history'):
        W_ee_h = data.W_ee_history[0]
        sampling_freq = shape(W_ee_h)[0]/(1.*data.c.N_steps[0])
        until_step = int(sampling_freq * data.c.steps_plastic[0])
        W_ee_h = W_ee_h[:until_step]
        nonzero = W_ee_h[0]!=0
        figure()
        plot(W_ee_h[:,nonzero][:,:100])
        xlabel('Step * %d'%(int(1./sampling_freq)))
        ylabel('Weight')
        utils.saveplot('Weighttraces_%s.%s'\
                        %(data.c.stats.file_suffix[0],ftype))
    
    if (data.__contains__('W_ee_history') and 
        data.__contains__('W_eu_history') and False):
        W_eu = data.W_eu_history[0][0]
        figure()
        plot(sum(abs(W_ee_h[:,nonzero][:-1]-W_ee_h[:,nonzero][1:]),1)
             /len(nonzero))
        xlabel('Step * %d'%(int(1./sampling_freq)))
        ylabel('Difference per synapse after %d steps'
                %(int(1./sampling_freq)))
        utils.saveplot('Weightdifftrace_%s.%s'\
                        %(data.c.stats.file_suffix[0],ftype))
                        
        figure()
        diffs = np.zeros(shape(W_ee_h))
        #TODO ugly brute force -> vectorize?!
        for t in range(shape(W_ee_h)[0]):
            for i in range(shape(W_ee_h)[1]):
                for j in range(i+1,shape(W_ee_h)[1]):
                    diffs[t,i,j] = diffs[t,j,i] = sum((W_ee_h[t,i,:]-
                                                       W_ee_h[t,j,:])**
                                                       2)
        all_diff_mean = mean(diffs,1).mean(1)
        within_input_mean = array([mean(diffs[:,W_eu[:,i]==1]
                                           [:,:,W_eu[:,i]==1],1).mean(1) 
                                    for i in range(shape(W_eu)[1])]
                                  ).mean(0)
        plot(all_diff_mean,label='overall')
        plot(within_input_mean,label='within input group')
        xlabel('Step * %d'%(int(1./sampling_freq)))
        ylabel('Average weight vector distance')
        legend(loc='best')
        utils.saveplot('Weightvectordiff_%s.%s'\
                        %(data.c.stats.file_suffix[0],ftype))
    
    ### Plot effect of double_synapses
    if (data.__contains__('W_ee_history') and
            data.__contains__('W_ee_2_history')):
        W_ee_h = data.W_ee_history[0]
        W_ee2_h = data.W_ee_2_history[0]
        until_step = int(shape(W_ee_h)[0]*(data.c.steps_plastic[0]
                                           /(1.*data.c.N_steps[0])))
        W_ee_h = W_ee_h[:until_step]
        W_ee2_h = W_ee2_h[:until_step]
        (meandiff, medcv, means, stds) = parallel_stats(W_ee_h,W_ee2_h)
        figure()
        x = linspace(0,data.c.N_steps[0],shape(meandiff)[0])
        plot(x,meandiff)
        ylabel('Mean weight difference')
        xlabel('Step')
        gca().locator_params(axis='x',nbins=4) # 4 ticks/axis
        xlim([x[0],x[-1]])
        tight_layout()
        utils.saveplot('DoubleSynapses_%s.%s'\
                        %(data.c.stats.file_suffix[0],ftype))
        
        nonzero = (W_ee_h[-1]!=0)*(W_ee2_h[-1]!=0)
        maxind = argmax(W_ee_h[:,nonzero][-1,:])
        minind = argmin(W_ee_h[:,nonzero][-1,:])
        randind = randint(shape(W_ee_h[:,nonzero][-1,:])[0])
        indices = [maxind, minind]
        figure()
        plot(x,W_ee_h[:,nonzero][:,maxind],label='max pair')
        plot(x,W_ee2_h[:,nonzero][:,maxind],'--k',label='max pair') 
        plot(x,W_ee_h[:,nonzero][:,randind],label='random pair')
        plot(x,W_ee2_h[:,nonzero][:,randind],':k',label='random pair') 
        #~ plt.gca().set_color_cycle(None)
        legend(loc='best')
        #~ ylabel('Weight strength')
        #~ tick_params(axis='x',labelbottom='off')
        #~ subplot(2,1,2)
        #~ plot(x,W_ee_h[:,nonzero][:,minind])
        #~ plot(x,W_ee2_h[:,nonzero][:,minind])  
        xlabel('Step')
        ylabel('Weight strength')
        tight_layout()
        utils.saveplot('DoubleSynapses_traces_%s.%s'\
                %(data.c.stats.file_suffix[0],ftype)) 

        figure()
        plot(x,medcv,label='SORN')
        plot([x[0],x[-1]],[0.083,0.083],'--k',label='Data')
        ylabel('Median CV between weight pairs')
        xlabel('Step')
        xlim([x[0],x[-1]])
        gca().locator_params(axis='x',nbins=4) # 4 ticks/axis
        legend(loc='best')
        tight_layout()
        utils.saveplot('DoubleSynapses_CV_%s.%s'\
                        %(data.c.stats.file_suffix[0],ftype))
                        
        figure()
        plot(means,stds/means,'o')
        # Exclude pairs with 0 stds 
        # (these are due to weights that hit the boundary and didn't 
        # receive perturbation in form of syn. failure yet
        means = means[stds>0]
        stds = stds[stds>0]
        import scipy
        (m,c,r,p,_) = scipy.stats.linregress(log10(means),log10(stds
                                                                /means))
        plot(means,10**c*means**m,'r', label='r=%.3f, p=%.3f'%(r,p))
        legend(loc='best')
        yscale('log')
        xscale('log')
        xlabel('Mean weight of pair')
        ylabel('CV')
        tight_layout()
        utils.saveplot('DoubleSynapses_weight_vs_CV_%s.%s'\
                        %(data.c.stats.file_suffix[0],ftype))
                        
        ### Plot total weights
        figure()
        plot(x,(W_ee_h+W_ee2_h).sum(2).mean(1))
        xlabel('Step')
        ylabel('Mean incomming weight')
        utils.saveplot('ScalingDevelopment_%s.%s'\
                %(data.c.stats.file_suffix[0],ftype)) 
                
    ### Plot development of lognormals
    if data.__contains__('W_ee_history'):
        W_ee_h = data.W_ee_history[0]
        until_step = int(shape(W_ee_h)[0]*(data.c.steps_plastic[0]
                                           /(1.*data.c.N_steps[0])))
        W_ee_h = W_ee_h[:until_step]
        figure()
        
        matchdiff = np.zeros(W_ee_h.shape[0])
        for i in range(W_ee_h.shape[0]):            
            logweight = W_ee_h[i][W_ee_h[i]>0]
            logbins = logspace(-2,0,10)
            (y,_) = histogram(logweight,bins=logbins)
            #fit data to lognormal
            x = logbins[:-1]+(logbins[0]+logbins[1])/2.0
            # Do the fitting
            # HACK: to avoid NaNs, this takes abs(var)
            def lognormal(x,mue,var,scale):
                return scale * (exp(- ((log(x)-mue)*(log(x)-mue)) /
                                (2*abs(var))) / (x*sqrt(2*pi*abs(var))))
            try:
                popt, pcov = curve_fit(lognormal, x, y)
                popt[1] = abs(popt[1])
                y_fitted = lognormal(x,*popt)
                matchdiff[i] = np.sqrt(np.sum((y_fitted-y)**2))
            except(RuntimeError):
                matchdiff[i] = -1
                
        x = linspace(0,data.c.N_steps[0],shape(matchdiff)[0])
        plot(x[matchdiff!=-1],matchdiff[matchdiff!=-1])
        xlabel('Step')
        ylabel('Deviation from lognormal fit')
        utils.saveplot('LogWeightDevelopment_%s.%s'\
                %(data.c.stats.file_suffix[0],ftype)) 
                
    ### Plot Development of STDP events
    if data.__contains__('W_ee.stdp_dw_history'):
        def moving_average(a, n=3) :
            ret = np.cumsum(a, dtype=float)
            ret[n:] = ret[n:] - ret[:-n]
            return ret[n - 1:] / n
            
        (tots,pots,deps) = data.__getattr__('W_ee.stdp_dw_history')[0].T
        tots = moving_average(tots,n=10)
        pots = moving_average(pots,n=10)
        deps = moving_average(deps,n=10)
        
        x = linspace(0,data.c.N_steps[0],shape(pots)[0])
        fig, ax1 = plt.subplots()
        ax1.plot(x,pots,'b')
        ax1.plot(x,deps,'r')
        ax1.set_ylabel('Number of events (mean-filtered)')

        ax2 = ax1.twinx()
        ax2.plot(x,tots,'g')
        # Double label necessary because legend captures only one axis
        ax2.plot([],[],'b',label='Potentiating events')
        ax2.plot([],[],'r',label='Depressing events')
        ax2.set_ylabel('Total change', color='g')
        for tl in ax2.get_yticklabels():
            tl.set_color('g')
            
        ax1.set_xlabel('Steps')
        ax2.legend(loc='lower right')
        tight_layout()
        utils.saveplot('Pot_vs_Dep_vs_time_%s.%s'\
                %(data.c.stats.file_suffix[0],ftype)) 
            
    ### Plot Multi-Dimensional-Scaling of Spont and Evoked activity
    if data.__contains__('SpontPattern') and plot_MDS:
        print 'plot MDS'
        N_samples = 150 # as in paper # 200 takes forever, 100 okay
        N_subs = 200
        shuffle_within = True # as in paper, shuffle within neuron
        # Use same data as PCA
        steps_plastic = data.c.steps_plastic[0]
        steps_noplastic_test = data.c.steps_noplastic_test[0]
        N_steps = steps_plastic + steps_noplastic_test
        input_spikes = data.Spikes[0][:,0:-steps_noplastic_test]
        spont_spikes = data.Spikes[0][:,-steps_noplastic_test:]
        input_index = data.InputIndex[0][0:-steps_noplastic_test]
        
        subs_neurons = arange(data.c.N_e[0])
        np.random.shuffle(subs_neurons)
        subs_neurons = subs_neurons[:N_subs]
        input_spikes = input_spikes[subs_neurons]
        spont_spikes = spont_spikes[subs_neurons]
        
        # Filter out empty states
        input_spikes = input_spikes[:,input_index != -1]
        input_index = input_index[input_index != -1]  
        
        # Avoid down_states
        spont_spikes = spont_spikes[:,sum(spont_spikes,0)!=0] 
        
        if data.c.stats.__contains__('only_last'):
            N_comparison = data.c.stats.only_last[0]\
                            -data.c.steps_noplastic_test[0]
        else:
            N_comparison = 2500
        assert(N_comparison > 0)
        assert(N_comparison <= steps_noplastic_test)
        maxindex = int(max(input_index))
        
        # Only use spikes that occured at the end of learning and spont
        last_input_spikes = input_spikes[:,-N_comparison:]
        last_spont_spikes = spont_spikes[:,-N_comparison:]
        last_input_index = input_index[-N_comparison:]
        # Filter out empty states
        last_input_spikes = last_input_spikes[:,last_input_index != -1]
        last_input_index = last_input_index[last_input_index != -1]
        
        # shuffle spontaneous spikes (as in Luczak2009)
        last_shuff_spikes = last_spont_spikes.copy()
        # this shuffles each row -> each neuron while keeping rate
        
        if shuffle_within:
            map(shuffle,last_shuff_spikes) # shuffle within neurons
        else:
            shuffle(last_shuff_spikes) # randomly change neurons
            
        
        def remove_duplicates(a):
            # from http://stackoverflow.com/questions/16970982/
            # find-unique-rows-in-numpy-array
            def unique_windows(ar):
                # adapted from:
                # https://github.com/numpy/numpy/blob/v1.8.0/numpy/lib/arraysetops.py#L93
                ar = ar.flatten()
                perm = ar.argsort(kind='quicksort')#kind='mergesort')
                aux = ar[perm]
                flag = np.concatenate(([True], aux[1:] != aux[:-1]))
                return aux[flag], perm[flag]
            
            b = np.ascontiguousarray(a).view(np.dtype((np.void, 
                                        a.dtype.itemsize * a.shape[1])))
            import platform
            if platform.system() is 'Windows':
                _, idx = unique_windows(b)
            else:
                _, idx = np.unique(b, return_index=True)
            return (a[idx], idx)
        
        # _nd = no_duplicates
        _, idx = remove_duplicates(last_input_spikes.T)
        last_input_spikes_nd = last_input_spikes[:,idx]
        last_input_index_nd = last_input_index[idx]
        _,idx = remove_duplicates(last_spont_spikes.T)
        last_spont_spikes_nd = last_spont_spikes[:,idx]
        _,idx = remove_duplicates(last_shuff_spikes.T)
        last_shuff_spikes_nd = last_shuff_spikes[:,idx]
        
        # MDS is highly time-intensive
        # -> only use N_samples spont and evoked states
        # To adjust to the plots of the paper, only a subset of stimuli
        # is selected
        N_sampleletters = 5
        
        # Sort input spikes by occurence in data to make sure that there
        # are enough samples for the N_sampleletters
        indices = arange(max(last_input_index_nd))
        index_freq = array([sum(last_input_index_nd==x) for x in 
                                  range(int(max(last_input_index_nd)))])
        indices_max = argsort(index_freq)[::-1]
        
        # Actually - just mix indices -> more random and if there aren't
        # enough samples, it will just take more letters into the plot
        shuffle(indices_max)
        
        sorted_spikes = hstack([last_input_spikes_nd[:,
                                where(last_input_index_nd==x)[0]
                                [:N_samples//N_sampleletters]] 
                                for x in indices_max])
        sorted_index  = hstack([last_input_index_nd[
                                where(last_input_index_nd==x)[0]
                                [:N_samples//N_sampleletters]] 
                                for x in indices_max])
        
        indices_1 = arange(shape(last_input_index_nd)[0])
        indices_2 = arange(shape(last_spont_spikes_nd)[1])
        indices_3 = arange(shape(last_shuff_spikes_nd)[1])
        shuffle(indices_1)
        shuffle(indices_2)
        shuffle(indices_3)
        
        last_input_spikes_nds = sorted_spikes[:,:N_samples]
        last_input_index_nds = sorted_index[:N_samples]
        
        last_spont_spikes_nds = last_spont_spikes_nd[:,
                                                  indices_2[:N_samples]]
        last_shuff_spikes_nds = last_shuff_spikes_nd[:,
                                                  indices_3[:N_samples]]

        
        import sklearn.manifold as skl_manifold
        import sklearn.metrics.pairwise as skl_metricsp
        import sklearn.preprocessing as skl_preprocessing
        
        import time
        tic = time.clock()
        
        # Shape for metrics and MDS.fit requires n_samples x n_features
        X = hstack((last_input_spikes_nds,last_spont_spikes_nds,
                    last_shuff_spikes_nds)).T
        
        # no effect, not mentioned in paper -> no
        normalize = False
        if normalize:
            X = skl_preprocessing.scale(X)
        
        dissimilarities = skl_metricsp.euclidean_distances(X)
        
        # manhattan more appropriate for spike trains?
        # more or less same thing -> use euclidean as in paper
        # although paper considered rates
        #~ dissimilarities = skl_metricsp.manhattan_distances(X)
        
        # Actual Multi-Dimensional Scaling taking place here
        
        matlab = use_matlab
        # No matlab on my windows machine
        if platform.system() is 'Windows':
            matlab = False
        if matlab:
            from mlabwrap import mlab
            # paper used nonmetrical MDS with stress1 crit. (default)
            X_t = mlab.mdscale(dissimilarities,2)
        else:     
            # This is performing _metrical_ mds (nonmetr. not working)
            mds = skl_manifold.MDS(n_components=2,#metric=False,
                                   dissimilarity='precomputed',n_jobs=1,
                                   max_iter=3000,eps=1e-12,verbose=True)
            X_t = mds.fit_transform(dissimilarities)
        
        tmp_patch_linewidth = mpl.rcParams['patch.linewidth']
        savefig_bbox = mpl.rcParams['savefig.bbox']
        #~ mpl.rcParams['savefig.bbox'] = 'tight'
        # This is reversed at end
        #~ mpl.rcParams['patch.linewidth'] = 0.0 
        fig = figure()
        ax = fig.add_subplot(111,aspect='equal')
        #~ ax2 = fig.add_axes([0.9, 0.1, 0.03, 0.8])
        #~ cb = mpl.colorbar.ColorbarBase(ax2, cmap=cmap, norm=norm, 
                        #~ spacing='proportional', ticks=bounds[:-1], 
                        #~ boundaries=bounds, format='%1i')
        msize = 40
        ax.scatter(X_t[-N_samples:,0],
                   X_t[-N_samples:,1],c='b',s=msize,linewidth=0,  
                   label='Shuff')
        ax.scatter(X_t[-N_samples*2:-N_samples,0],
           X_t[-N_samples*2:-N_samples,1],c='k',s=msize,linewidth=0,
           label='Spont')
        index_set = set(last_input_index_nds)
        for (k,i) in enumerate(index_set):
            indices = where(last_input_index_nds==i)[0]
            N_ind = shape(indices)[0]
            if k==0:
                ax.scatter(X_t[indices,0],X_t[indices,1],label='Evoked',
                                    c='r',s=msize,linewidth=0)
                                    #ones(N_ind)*i,cmap=cmap,norm=norm)
            else:
                ax.scatter(X_t[indices,0],X_t[indices,1],
                                    c='r',s=msize,linewidth=0)
                                    #ones(N_ind)*i,cmap=cmap,norm=norm)

        #~ ax.xaxis.set_visible(False)
        #~ ax.yaxis.set_visible(False)
        #~ ax2.set_ylabel('Input Index', size=12)
        # this assumes that spontpattern exists
        #~ ax2.set_yticklabels(array([x for x in words_subscript]))
        # reverse linewidth for legend
        mpl.rcParams['patch.linewidth'] = tmp_patch_linewidth
        ax.legend()
        toc = time.clock()
        print toc-tic
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_xlabel('MDS dimension 1')
        ax.set_ylabel('MDS dimension 2')
        tight_layout()
        eps = (max(X_t[:,0])-min(X_t[:,0]))/20.
        xlim([min(X_t[:,0])-eps,max(X_t[:,0])+eps])
        eps = (max(X_t[:,1])-min(X_t[:,1]))/20.
        ylim([min(X_t[:,1])-eps,max(X_t[:,1])+eps])
        utils.saveplot('SpontMDS_%s.%s'\
                        %(data.c.stats.file_suffix[0],ftype))
        #~ mpl.rcParams['savefig.bbox'] = savefig_bbox                
        ## Now compute closest spont. and shuffled state for each evoked
        
        N_samples = shape(last_input_index)[0]
        D_spont = np.zeros(N_samples)
        D_shuff = np.zeros(N_samples)
        
        # Minimal distance between vector y and rows in X
        def min_dist(X,y):
            #~ return min(sqrt(sum((X-y)**2,1)))
            return min(sum(abs(X-y),1))
        
        for i in range(N_samples):
            D_spont[i] = min_dist(last_spont_spikes.T,
                                  last_input_spikes[:,i])
            D_shuff[i] = min_dist(last_shuff_spikes.T,
                                  last_input_spikes[:,i])
        # Test if difference is significant
        from scipy import stats
        _,p = stats.wilcoxon(D_shuff-D_spont)
        
        figure()
        ax = fig.add_subplot(111,aspect='equal')
        scatter(D_spont,D_shuff,color='k',
                            label='Manhattan distance to closest state')
        hold('on')
        x_lim = xlim()
        y_lim = ylim()
        max_xy = max([x_lim[1],y_lim[1]])
        min_xy = min([x_lim[0],y_lim[0]])
        lim = [min_xy,max_xy]
        plot(lim,lim,'r--')
        xlim(lim)
        ylim(lim)
        xlabel('D_spont')
        ylabel('D_shuff')
        legend(loc='lower right')
        #~ title('Manhattan distance to closest point')
        tight_layout()
        axis('equal')
        utils.saveplot('SpontShuffScatter_%s.%s'\
                        %(data.c.stats.file_suffix[0],ftype))
        
        figure()
        hist(D_shuff-D_spont)
        hold('on')
        y_lim = ylim()
        plot([0,0],[0,y_lim[1]],'r',linewidth=3)
        xlabel('D_shuff - D_spont')
        ylabel('Events #')
        #~ title('Wilcoxon_p = %.3f'%p)
        tight_layout()
        utils.saveplot('SpontShuffHist_%s.%s'\
                        %(data.c.stats.file_suffix[0],ftype))
                                
    if data.__contains__('patternprobability'):
        def KL(p,q):
            # in case zero-correction was deactivated
            q = q[p>0]
            p = p[p>0]
            p = p[q>0]
            q = q[q>0]
            q /= sum(q)
            p /= sum(p)
            kl = sum(p*log2(p/q))
            return kl
        
        p_evoked_1 = data.patternprobability[0][0]
        p_evoked_2 = data.patternprobability[0][1]
        p_spont_1 = data.patternprobability[0][2]
        p_spont_2 = data.patternprobability[0][3]
        p_evoked = (p_evoked_1+p_evoked_2)/2
        p_spont = (p_spont_1+p_spont_2)/2
        
        kl_evoked_spont = KL(p_evoked,p_spont)
        kl_spont_evoked = KL(p_spont,p_evoked)
        
        kl_evoked_12 = KL(p_evoked_1,p_evoked_2)
        kl_evoked_21 = KL(p_evoked_2,p_evoked_1)
        kl_spont_12 = KL(p_spont_1,p_spont_2)
        kl_spont_21 = KL(p_spont_2,p_spont_1)
        
        figure()
        bar([1,2,3,4],[kl_spont_evoked,kl_evoked_spont,kl_evoked_12,
            kl_spont_12],align='center')
        xticks([1,2,3,4],['$D(s||e)$','$D(e||s)$','$D(e||e)$',
            '$D(s||s)$'])
        ylabel('KL-Divergence')
        xlim([0.5,4.5])
        utils.saveplot('KLdiv_%s.%s'\
                        %(data.c.stats.file_suffix[0],ftype))
                        
        figure()
        N_bars = 51
        bar(arange(N_bars),p_evoked[:N_bars],log=True,align='center',
            color='b',alpha=0.5,label='evoked')
        bar(arange(N_bars),p_spont[:N_bars],log=True,align='center',
            color='r',alpha=0.5,label='spont')
        xlim([-1,N_bars])
        xlabel('Pattern number')
        ylabel('Pattern frequencies')
        legend(loc='best')
        utils.saveplot('KLdiv_bars_%s.%s'\
                        %(data.c.stats.file_suffix[0],ftype))

    ### Plot Activity Stats
    if data.__contains__('activity'):
        print 'plot activity'
        figure()
        if data.c.N_steps[0]>1000000:
            y = data.activity[0][::100]
            x = linspace(0,data.c.N_steps[0],shape(y)[0])
            plot(x,y)
        else:
            plot(data.activity[0])
        #~ title('Activity over time')
        xlabel('Simulation Step')
        ylabel('Mean Activity')
        tight_layout()
        utils.saveplot('Activity_%s.%s'
                        %(data.c.stats.file_suffix[0],ftype))

        if data.c.steps_plastic[0]>0:
            figure()
            last_steps = 1000
            steps = data.c.steps_plastic[0]
            plot(data.activity[0][steps-last_steps:steps-1], 
                         data.activity[0][1+steps-last_steps:steps],'.')
            plot([0,1],[0,1],'-')
            xlim([0,1])
            ylim([0,1])
            #~ title('Change of Activity')
            xlabel('$\overline{X}(t)$')
            ylabel('$\overline{X}(t+1)$')
            tight_layout()
            utils.saveplot('ActivityChange_%s.%s'%\
                                    (data.c.stats.file_suffix[0],ftype))

    if data.__contains__('meanactivity'):
        test_words = data.c.source.test_words[0]
        baseline = data.meanactivity[0][0]
        act = {}
        act_2nd = {}
        start = 1
        for word in test_words:
            length = len(word)
            act[word] = mean(data.meanactivity[0][start:start+length])
            act_2nd[word] = data.meanactivity[0][start+1]
            start += length
        # Colors from figures from paper
        c_gray = '#929496'
        c_blue = '#33348e'
        c_red  = '#cc2229'
        c_green= '#33a457'
        col = {'ABCD':c_blue,'DCBA':c_red,'A_CD':c_red,'E_CD':c_green}
                
        if data.c.source.control:
            condition = 'Control'
        else:
            condition = 'Experimental'
        
        figure()
        start = 1
        for word in test_words:
            length = len(word)
            plot(data.meanactivity[0][start:start+length],c=col[word],
                 label=word)
            start += length
        xlabel(condition)
        ylabel('Magnitude')
        legend(loc='best')
        utils.saveplot('Mean_time_%s.%s'%\
                                    (data.c.stats.file_suffix[0],ftype))
        figure()
        bar(1,baseline,color=c_gray,label='Baseline',align='center')
        for (i,word) in enumerate(test_words):
            bar(i+2,act[word],color=col[word],label=word,align='center')
        tick_params(axis='x',which='both',bottom='off',top='off',
                    labelbottom='off')
        xlim([0.5,i+2.5])
        xlabel(condition)
        ylabel('Sequence magnitude')
        legend(loc='upper left')
        utils.saveplot('Mean_reverse_%s.%s'%\
                                    (data.c.stats.file_suffix[0],ftype))
        figure()
        for (i,word) in enumerate(test_words):
            bar(i+1,act[word],color=col[word],label=word,align='center')
        l = i+1
        for (i,word) in enumerate(test_words):
            bar(i+2+l,act_2nd[word],color=col[word],align='center')
        legend(loc='lower left')
        tick_params(axis='x',which='both',bottom='off',top='off')
        xticks([i//2+1,l+3],['Full sequence','Second element'])
        xlim([0,2*(i+1)+2])
        ylabel('Magnitude')
        utils.saveplot('Mean_2nd_%s.%s'%\
                                    (data.c.stats.file_suffix[0],ftype))
        
    ### Plot endweight Stat
    if data.__contains__('endweight'):
        N_e = data.c.N_e[0]
        N_u_e = data.c.N_u_e[0]
        #~ mean_outgoing = np.zeros(int(N_e//N_u_e))
        #~ for i in range(6):
            #~ if i <= 5:
                #~ tmp = (data.endweight[0]\
                                #~ [:,i*N_u_e:i*N_u_e+N_u_e]).flatten()
                #~ tmp = tmp[tmp>0]
            #~ else:
                #~ tmp = (data.endweight[0][:,i*N_u_e:]).flatten()
                #~ tmp = tmp[tmp>0]
            #~ mean_outgoing[i] = mean(sort(tmp)[-100:])
        #~ figure()
        #~ bar(arange(shape(mean_outgoing)[0]),mean_outgoing,\
                                                    #~ align='center')
        #~ xticks([0,1,2,3,4,5],['A','B','N','M','X','noinput'])
        #~ ylabel('Mean outgoing weight')
        #~ xlabel('Input population')
        #~ tight_layout()
        #~ utils.saveplot('ProjWeights_%s.pdf'%\
                                    #~ (data.c.stats.file_suffix[0]))
        
        print 'plot endweight'
        # First the logweight:
        logweight = data.endweight[0][data.endweight[0]>0]
        figure()
        logbins = logspace(-2,0,10)
        (y,_) = histogram(logweight,bins=logbins)
        #fit data to lognormal
        x = logbins[:-1]+(logbins[0]+logbins[1])/2.0
        semilogx(x,y,'.')
        hold('on')

        # Do the fitting
        # HACK: abs(var)
        def lognormal(x,mue,var,scale):
            return scale * (exp(- ((log(x)-mue)*(log(x)-mue)) /
                                (2*abs(var))) / (x*sqrt(2*pi*abs(var))))

        try:
            popt, pcov = curve_fit(lognormal, x, y)
            popt[1] = abs(popt[1])
        except(RuntimeError):
            popt = [0,0,0]
        curve_x = logspace(-2,0,100)
        fitted_y = lognormal(curve_x,*popt)
        semilogx(curve_x,fitted_y)
        #~ title('Final Weight Distribution')
        xlabel('Weight')
        ylabel('Frequency')
        legend(('Data', 'Logn. fit'),loc='best')
        tight_layout() # throws errors
        utils.saveplot('LogWeights_%s.%s'%\
                                    (data.c.stats.file_suffix[0],ftype))
        
        # Now scale-free property
        tmp = data.endweight[0]>0.0+0.0
        binary_connections = tmp+0.0
        in_degree = sum(binary_connections,1)
        out_degree = sum(binary_connections,0)
        fig = figure()
        fig.add_subplot(131)
        hist(in_degree)
        ylabel('Frequency')
        xlabel('In degree')
        fig.add_subplot(132)
        hist(out_degree)
        xlabel('Out degree')
        fig.add_subplot(133)
        hist(in_degree+out_degree)
        xlabel('In+out degree')
        plt.suptitle('Degree distributions')
        utils.saveplot('Degree_Distributions_%s.%s'%\
                                    (data.c.stats.file_suffix[0],ftype))
    
    # Plot response probabilities of input units                       
    if (data.__contains__('Spikes') and data.__contains__('InputUnits')
        and data.__contains__('Bayes')):
        steps_plastic = data.c.steps_plastic[0]
        steps_noplastic_train = data.c.steps_noplastic_train[0]
        steps_noplastic_test = data.c.steps_noplastic_test[0]
        input_w = data.__getattr__('W_eu.W_history')[0][0]
        spikes = data.Spikes[0][:,steps_plastic:
                                  steps_plastic+steps_noplastic_train]
        inputs = data.InputIndex[0][steps_plastic:
                                    steps_plastic+steps_noplastic_train]
        N_inputs = int(inputs.max()+1)
        if any(inputs==-1):
            N_inputs += 1
        probs = zeros((N_inputs,data.c.N_e[0]))
        count = zeros(N_inputs)
        for i in range(len(inputs)):
            probs[inputs[i]] += spikes[:,i]
            count[inputs[i]] += 1.
        for i in range(N_inputs):
            probs[i]/=count[i]
            
        # Assume that in bayes training, only two words with identical 
        # length
        A_probs = probs[0]
        if any(inputs==-1):
            B_probs = probs[(N_inputs-1)//2]
        else:
            B_probs = probs[N_inputs//2]
            
        filename = os.path.join(pickle_dir,"source_plastic.pickle")
        source = pickle.load(gzip.open(filename,"r"))
        if isinstance(source,TrialSource):
            source = source.source
            
        assert(source.words[1][0]=='B' and len(source.words)==2)
                    
        A_units = input_w[:,source.lookup['A']]
        B_units = input_w[:,source.lookup['B']]
        # First letter is unit, second letter is presented letter
        AA_probs = A_probs[where(A_units==1)[0]]
        BB_probs = B_probs[where(B_units==1)[0]]
        BA_probs = A_probs[where(B_units==1)[0]]
        AB_probs = B_probs[where(A_units==1)[0]]
        
        figure()
        x = arange(len(AA_probs))+1
        subplot(2,2,1)
        bar(x,AA_probs,align='center')
        xlim([0.5,10.5])
        ylim([0,1])
        ylabel('A presented')
        subplot(2,2,2)
        bar(x,BA_probs,align='center')
        xlim([0.5,10.5])
        ylim([0,1])
        subplot(2,2,3)
        bar(x,AB_probs,align='center')       
        xlim([0.5,10.5])
        ylim([0,1])
        ylabel('B presented')
        xlabel('A units')
        subplot(2,2,4)
        bar(x,BB_probs,align='center')
        xlim([0.5,10.5])
        ylim([0,1])
        xlabel('B units')
                
        utils.saveplot('response_prob_%s.%s'%\
                                    (data.c.stats.file_suffix[0],ftype))
              
    if data.__contains__('smallworld'):
        figure()
        for item in data.smallworld:
            plot([1,2,3],item[0],'o')
        plot([0,4],[1,1],'--')
        xticks([1,2,3],['gamma','lambda','S_W'])
        title('small-world-ness')
        utils.saveplot('small_world_%s.%s'%\
                                    (data.c.stats.file_suffix[0],ftype))

    ### Plot ISIs
    if data.__contains__('ISIs'):
        print 'plot ISIs'
        figure()
        ISIs = data.ISIs[0]
        if shape(ISIs)[1] > 50:
            ISIs = ISIs[:,:50]
        
        x = np.array(range(0,shape(ISIs)[1]))
        y = ISIs[randint(0,shape(ISIs)[0])]
        

        # Do the fitting
        def exponential(x, a, b):
            return a * np.exp(-b*x)
        if data.c.stats.__contains__('ISI_step') \
           and data.c.stats.ISI_step > 1:
            start = int(round(mean(argmax(ISIs,1))))
            x_fit = x[start::data.c.stats.ISI_step[0]]
            y_fit = y[start::data.c.stats.ISI_step[0]]
        else:
            x_fit = x
            y_fit = y
        popt, pcov = curve_fit(exponential, x_fit, y_fit)
        x = np.array(range(shape(ISIs)[1]))
        fitted_y = exponential(x,*popt)
        if data.c.stats.__contains__('ISI_step') \
           and data.c.stats.ISI_step[0] > 1:
            plot(x[start:],y[start:], '.')
            plot(x[start:],fitted_y[start:])
            xlim([start-1,max(x)])
        else:
            plot(x,y, '.')
            plot(x,fitted_y)
            xlim([-1,shape(ISIs)[1]])
        #~ x_lim[0] -= 1
        #~ xlim(x_lim)

        #~ title('Interspike Intervals')
        xlabel('ISI')
        ylabel('Frequency')
        if data.c.stats.__contains__('ISI_step')\
           and data.c.stats.ISI_step[0] > 1:
            fitlabel = 'Exp. fit (sampling=%d)'%data.c.stats.ISI_step[0]
        else:
            fitlabel = 'Exp. fit'
        legend(('Data', fitlabel))
        tight_layout()
        utils.saveplot('ISIs_%s.%s'%(data.c.stats.file_suffix[0],ftype))
        
        # Plot the CV hist
        figure()
        CVs = []
        intervals = arange(shape(ISIs)[1])
        for i in range(shape(ISIs)[0]):
            tmp = repeat(intervals.tolist(),ISIs[i].tolist())
            CVs.append(std(tmp)/mean(tmp))
        CVs = array(CVs)
        CVs = CVs[CVs>0] #ignore nans
        n, bins, patches = hist(CVs)
        xlim([floor((min(bins[n>0])-0.1)*10)*0.1, 
                                    floor((max(bins[n>0])+0.2)*10)*0.1])
        xlabel('ISI CV')
        ylabel('Frequency')
        utils.saveplot('ISIs_CV_%s.%s'%(data.c.stats.file_suffix[0],
                                        ftype))

    ### Plot ConnectionFraction
    if data.__contains__('ConnectionFraction'):
        print 'plot connectionfraction'
        figure()
        plot(data.ConnectionFraction[0][:data.c.steps_plastic[0]])
        #~ title('Fraction of Exc-Exc Connections')
        xlabel('Step')
        ylabel('Fraction of E-E connections')
        tight_layout()
        utils.saveplot('Connections_%s.%s'%\
                                    (data.c.stats.file_suffix[0],ftype))

    ### Plot WeightLifetime
    if data.__contains__('WeightLifetime') and \
        any(data.WeightLifetime[0][:] > 0):
            
        print 'plot weightlifetime'
        figure()
        logbins = logspace(2,4,20)
        (y,_) = histogram(data.WeightLifetime[0]\
                                [data.WeightLifetime[0]>0],bins=logbins)
        x = logbins[:-1]+(logbins[0]+logbins[1])/2.0
        loglog(x,y,'.')
        def powerlaw(x,a,k):
            return a*x**k
        popt, pcov = curve_fit(powerlaw, x, y)
        fitted_y = powerlaw(x,*popt)
        plot(x,fitted_y)

        title('Weight Lifetime (%s)'%(data.c.stats.file_suffix[0]))
        xlabel('Lifetime (Steps)')
        ylabel('Frequency')
        legend(('data','powerlaw-fit (a=%.3f k=%.3f)'%
                       (popt[0],popt[1])), loc='best')
        tight_layout()
        utils.saveplot('WeightLifetime_%s.%s'%\
                                    (data.c.stats.file_suffix[0],ftype))


    ### Plot WeightChangeStat
    if data.__contains__('WeightChange'):
        print 'plot weightchange'
        # 0:weights, 1:abschange, 2:relchange
        fig = figure()
        fig.add_subplot(211)
        plot(data.WeightChange[0][0][-(data.WeightChange[0][0]==0)],
             data.WeightChange[0][1][-(data.WeightChange[0][0]==0)],'.')
        ylabel('Absolute Change')
        fig.add_subplot(212)
        plot(data.WeightChange[0][0][-(data.WeightChange[0][0]==0)],
             data.WeightChange[0][2][-(data.WeightChange[0][0]==0)],'.')
        xlabel('Weight')
        ylabel('Relative Change')        
        plt.suptitle('Change of Weights over %d Steps (%s)'%\
                                     (3000,data.c.stats.file_suffix[0]))
        tight_layout()
        utils.saveplot('WeightChange_%s.%s'%\
                                    (data.c.stats.file_suffix[0],ftype))
        
    ### Plot InputWeightStat
    if data.__contains__('InputWeight'):
        print 'plot InputWeight'
        figure()
        #samples = shape(data.InputWeight[0])[2]
        #sums_weights = np.zeros((samples))
        #total_weights = np.zeros((samples))
        #for i in range(samples):
        #    sums_weights[i] = sum(data.InputWeight[0][:,:,i])
        #    total_weights[i] = sum(data.InputWeight[0][:,:,i]>0)
        sums_weights = data.InputWeight[0].sum(0).sum(0)
        plot(sums_weights)
        xlabel('Step')
        ylabel('Sum of all input weights')
        title('Input Weight Influence')
        
        total_weights = (data.InputWeight[0]>0).sum(0).sum(0)
        figure()
        plot(total_weights)
        xlabel('Step')
        ylabel('Number of input connections')
        title('Input Weight Influence')
        tight_layout()
        utils.saveplot('InputWeight_%s.%s'%\
                                    (data.c.stats.file_suffix[0],ftype))
    
    #~ ### Plot the spikes (ugly - zooming necessary-unsuited for print)
    if data.__contains__('Spikes') and plot_spikes:
        print 'plot Spikes'
        spikes = data.Spikes[0]
        inputs = data.InputIndex[0]

        if data.__contains__('W_eu.W_history') and False:
            # Put input units to top for easier comparison with Andreeas
            # data
            input_w = data.__getattr__('W_eu.W_history')[0][0]
            units = sum(input_w,1)
            N_e = shape(input_w)[0]
            all_units = range(N_e)
            start = 0
            print shape(input_w)[0]
            for i in range(shape(input_w)[1]):
                toswap = [x for x in where(input_w[:,i]==1)[0] if x in \
                                                              all_units]
                end = start+len(toswap)
                tmp = spikes[start:end,:]
                spikes[start:end,:] = spikes[toswap,:]
                spikes[toswap,:] = tmp
                for item in toswap:
                    all_units.remove(item)
                start = end
        figure()
        steps = -1#data.c.steps_plastic[0]
        last_n_spikes = 200
        spikes_small = spikes[:,steps-last_n_spikes:steps]
        inputs_small = inputs[steps-last_n_spikes:steps]
        for (i,sp) in enumerate(spikes_small):
            s_train = where(sp==1)[0]
            plot(s_train,np.ones(s_train.shape)*i,'o',color='k',
                 markersize=1)
            hold('on')
            
        # Plot input presentation area
        presented = arange(last_n_spikes)[inputs_small!=-1]
        skips = arange(len(presented))[(presented[1:]-presented[:-1])>1]
        if len(skips)>1: # -> more than just spont. or evoked activity
            start = presented[0]
            for s in skips: 
                axvspan(start-0.5,presented[s]+0.5,color='#E6E6E6') 
                start = presented[s+1]
            axvspan(start-0.5,presented[-1],color='#E6E6E6') 


        xlabel('Step')
        ylabel('Excitatory Neuron')
        xlim([0,last_n_spikes])
        tight_layout()
        utils.saveplot('Spikes_small_%s.%s'
                        %(data.c.stats.file_suffix[0],ftype))
                        
        matshow(spikes,aspect='auto')
        xlabel('Step')
        ylabel('Neuron')
        #~ utils.saveplot('Spikes_full_%s.%s'
                          #~ %(data.c.stats.file_suffix[0],ftype))
        
    if (data.__contains__('Spikes') and data.__contains__('endweight') 
        and data.__contains__('SpontIndex')):
        # This analyzes the order of neurons
        spikes = data.Spikes[0]
        inputs = data.InputIndex[0]
        steps = data.c.steps_plastic[0]+data.c.steps_noplastic_train[0]
        last_n_spikes = 2500
        padding_b = 5 # padding before stimulus onset
        padding_a = 5 # padding after stimulus offset
        filename = os.path.join(pickle_dir,"source_train.pickle")
        source = pickle.load(gzip.open(filename,"r"))
        if isinstance(source,TrialSource):
            source = source.source
        max_word_length = max([len(x) for x in source.words])
        N_words = len(source.words)
        words = source.words
        spikes_last = spikes[:,steps-last_n_spikes:steps]
        inputs_last = inputs[padding_b+steps-last_n_spikes:
                             steps-padding_a-max_word_length]
        N_e = data.c.N_e[0]
        
        spikes_mean = np.zeros((N_words+1, N_e,max_word_length
                                +padding_b+padding_a))
        word_index = 0
        
        words = words+[' '*max_word_length]
        N_words+=1
        figure()
        for (w_i,w) in enumerate(words):
            # Add padding here to switch to spiking reference frame
            if w_i == N_words-1:
                start_letter = 2
                padding_b += start_letter
                padding_a -=start_letter
                # spont case
                spikes_last = spikes[:,-last_n_spikes:]
                #~ start_indices = arange(padding,last_n_spikes-padding
                                    #~ -max_word_length,max_word_length)
                assert(len(data.SpontIndex[0] >= last_n_spikes))
                sponts_last = data.SpontIndex[0][padding_b:-padding_a
                                                 -max_word_length]
                start_indices = where(sponts_last==start_letter)[0]\
                                +padding_b
                word_len = max_word_length
            else:
                start_indices = where(inputs_last==word_index)[0]\
                                +padding_b
                word_index += len(w)
                word_len = len(w)
            for i in range(-padding_b,max_word_length+padding_a):
                spikes_mean[w_i,:,i+padding_b] = \
                                mean(spikes_last[:,start_indices+i],1)
            if w_i==0:
                sorted_indices = argsort((spikes_mean[w_i,:,\
                               padding_b:padding_a+word_len]).argmax(1))    
            
            #~ fig = figure(figsize=(7,6))
            subplot(N_words,1,w_i+1)
            imshow(spikes_mean[w_i,sorted_indices],vmin=0,vmax=1,
                   aspect=word_len/float(N_e),
                   interpolation='None')
            ax = gca()
            ax.xaxis.set_ticks_position('bottom')
            padding = (padding_b+padding_a)//2
            x_ticks = arange(-1,word_len+1)+padding
            x_ticks[0] = padding//2
            x_ticks[-1] = padding//2 + max_word_length + padding
            x_labels = ['Before'] + [x for x in w] + ['After']
            ax.set_xticks(x_ticks)
            ax.set_xticklabels(x_labels)        
            #~ ax.set_title('Average activity sorted by %s'%
                                                #~ source.words[0])
            ax.set_ylabel('Neuron')
            if w_i==N_words-1:
                ax.set_xlabel('Letter')
        utils.saveplot('Sorted_spikes_%s_%s.%s'%
                                (w,data.c.stats.file_suffix[0],ftype))

        # Rearrange weight matrix according to sorting
        weights = data.endweight[0]
        for i in range(N_e):
            weights[i] = weights[i,sorted_indices]
        weights = weights[sorted_indices]
        matshow(weights)
        xlabel('From')
        ylabel('To')
        title('Rearranged weights for word %s'%source.words[0])
        utils.saveplot('Sorted_weights_%s.%s'%
                                    (data.c.stats.file_suffix[0],ftype))
        
    if data.__contains__('SpikesInh'):
        activity_inh = sum(data.SpikesInh[0],0)/float(data.c.N_i[0])
        activity_exc = sum(data.Spikes[0],0)/float(data.c.N_e[0])
        
        # Adapted from http://matplotlib.org/examples/pylab_examples/scatter_hist.html
        left, width = 0.1, 0.65
        bottom, height = 0.1, 0.65
        bottom_h = left_h = left+width+0.02

        rect_scatter = [left, bottom, width, height]
        rect_histx = [left, bottom_h, width, 0.2]
        rect_histy = [left_h, bottom, 0.2, height]
        rect_bar = [left_h, bottom_h, 0.2, 0.2]

        # start with a rectangular Figure
        figure(figsize=(8,8))
        #~ plt.figure(1, figsize=(8,8))

        axScatter = plt.axes(rect_scatter)
        axHistx = plt.axes(rect_histx)
        axHisty = plt.axes(rect_histy)
        axBar = plt.axes(rect_bar)
        from matplotlib.ticker import NullFormatter
        nullfmt = NullFormatter()
        axHistx.xaxis.set_major_formatter(nullfmt)
        axHisty.yaxis.set_major_formatter(nullfmt)
        import matplotlib.ticker as plticker
  
        axScatter.scatter(activity_exc,activity_inh)
        axScatter.set_xlabel('Relative excitation')
        axScatter.set_ylabel('Relative inhibition')
        axScatter.set_xlim((0,min(axScatter.get_xlim()[1],1)))
        axScatter.set_ylim((0,min(axScatter.get_ylim()[1],1)))
        binsx = np.arange(0,axScatter.get_xlim()[1],
                          axScatter.get_xlim()[1]/float(20))
        binsy = np.arange(0,axScatter.get_ylim()[1],
                          axScatter.get_ylim()[1]/float(20))
        axHistx.hist(activity_exc, bins=binsx)
        axHisty.hist(activity_inh, bins=binsy, orientation='horizontal')
        axHistx.set_xlim(axScatter.get_xlim())
        axHisty.set_ylim(axScatter.get_ylim())
        # these locators puts ticks at regular intervals
        locx = plticker.MultipleLocator(base=axHistx.get_ylim()[1]//2) 
        locy = plticker.MultipleLocator(base=axHisty.get_xlim()[1]//2)
        axHistx.yaxis.set_major_locator(locx)
        axHisty.xaxis.set_major_locator(locy)
        
        axBar.barh([0.5,1.5],
            [sum(sum(data.SpikesInh[0][:,-1000:],1)==0)
             /float(data.c.N_i[0]),
             sum(sum(data.SpikesInh[0][:,-1000:],1)==1000)
             /float(data.c.N_i[0])], align='center')
        plt.sca(axBar)
        plt.xlim([0,1])
        plt.ylim([0,2])
        plt.text(0,0.5,'Nonactive',verticalalignment='center')
        plt.text(0,1.5,'Nonsilent',verticalalignment='center')
        plt.xticks([0,1])
        plt.yticks([])
        
        utils.saveplot('ExIn_%s.%s'%(data.c.stats.file_suffix[0],ftype))
        
        figure()
        steps = data.c.steps_plastic[0]
        last_n_spikes = 200
        spikes_small = data.SpikesInh[0][:,steps-last_n_spikes:steps]
        for (i,sp) in enumerate(spikes_small):
            s_train = where(sp==1)[0]
            plot(s_train,np.ones(s_train.shape)*i,'o',color='k',
                 markersize=1)
            hold('on')
        ylim([0,data.c.N_i[0]])
        xlabel('Step')
        ylabel('Inhibitory neuron')
        tight_layout()
        utils.saveplot('Spikes_small_Inh_%s.%s'
                       %(data.c.stats.file_suffix[0],ftype))
    
    ### Plot the full endweight
    if data.__contains__('FullEndWeight'):
        print 'plot FullEndWeight'
        matshow(data.FullEndWeight[0])
        title('FullEndWeight')
        utils.saveplot('FullEndWeight_%s.%s'
                       %(data.c.stats.file_suffix[0],ftype))
        
    ### Save spikes and weights for Lizier analysis
    if data.__contains__('Spikes') \
        and data.__contains__('FullEndWeight'):
        path = '/home/chartmann/Desktop/Organisationsstruktur/sorn_'+\
                datetime.datetime.now().strftime('%d-%m-%Y_%H-%M-%S')
        savemat(path,
                mdict={'W_EE':data.FullEndWeight[0],\
                       'W_EI':getattr(data,'W_ei.W_history')[0][0],\
                       'W_IE':getattr(data,'W_ie.W_history')[0][0],\
                       'W_EU':getattr(data,'W_eu.W_history')[0][0],\
                       'T_e':data.T_e_history[0],\
                       'T_i':data.T_i_history[0][0],\
                       'spikes_exc':data.Spikes[0].T,\
                       'spikes_inh':data.SpikesInh[0].T,\
                       'input_index':data.InputIndex[0]})
        savez(path,
              W_EE=data.FullEndWeight[0],
              W_EI=getattr(data,'W_ei.W_history')[0][0],
              W_IE=getattr(data,'W_ie.W_history')[0][0],
              W_EU=getattr(data,'W_eu.W_history')[0][0],
              T_e=data.T_e_history[0],
              T_i=data.T_i_history[0][0],
              spikes_exc=data.Spikes[0].T,
              spikes_inh=data.SpikesInh[0].T,
              input_index=data.InputIndex[0])
    
    ### Plot SpontPatterns    
    if data.__contains__('SpontPattern'):
        print 'plot SpontPattern'
        filename = os.path.join(pickle_dir,"source_plastic.pickle")
        source_plastic = pickle.load(gzip.open(filename,"r"))
        if isinstance(source_plastic,TrialSource):
            source_plastic = source_plastic.source
            
        steps_noplastic_test = data.c.steps_noplastic_test[0]
            
        words = source_plastic.words
        words_subscript = ["$\mathrm{%s}_{%i%i}$"%(
                            ("\%s" if letter == "_" else letter),i,j) \
                            for (i,word) in enumerate(words) \
                            for (j,letter) in enumerate(word)]
        words_subscript = [letter for word in words for letter in word]
        figure()
        patternfreqs = data.SpontPattern[0]/(1.*data.NormLast[0])
        patternfreqs[1,:] /= sum(patternfreqs[1,:])
        bar(arange(shape(patternfreqs)[1]),patternfreqs[0,:],\
            align='center')#,color=repeat(['b','r'],[4,4]))
        ax = gca()
        ax.set_xticks(arange(len(words_subscript)))
        ax.set_xticklabels(array([x for x in words_subscript]))
        #~ title('CharFrequencies')
        ylabel('Relative Frequency')
        xlabel('Letter')
        tight_layout()
        utils.saveplot('SpontFreq_%s.%s'%(data.c.stats.file_suffix[0],
                        ftype))
        
        figure()
        bar(arange(shape(patternfreqs)[1]),patternfreqs[1,:],\
            align='center')
        ax = gca()
        ax.set_xticks(arange(len(words)*2))
        ax.set_xticklabels(words + [x[::-1] for x in words],rotation=30)
        #~ title('Pattern Frequencies')
        ylabel('Frequency')
        tight_layout()
        utils.saveplot('SpontPatterns_%s.%s'\
                        %(data.c.stats.file_suffix[0],ftype))

                        
    if data.__contains__('SpontTransition'):
        print 'plot SpontTransition'
        transitions = data.SpontTransition[0]
        for i in range(shape(transitions)[0]):
            transitions[:,i] /= sum(transitions[:,i])# normalize
        figure()
        im = imshow(transitions,interpolation='none',vmin=0, 
                                                 vmax=1)
        xlabel('From')
        ylabel('To')
        ax = gca()
        ax.set_xticks(arange(len(words_subscript)))
        ax.set_xticklabels(array([x for x in words_subscript]))
        ax.set_yticks(arange(len(words_subscript)))
        ax.set_yticklabels(array([x for x in words_subscript]))
        colorbar(im, use_gridspec=True)
        tight_layout()
        utils.saveplot('Spont_transitions_%s.%s'\
                        %(data.c.stats.file_suffix[0],ftype))
                        
        # Predict spontpatterns from transitions
        av_trans = np.zeros(len(words)*2)
        w_i = 0
        for (i,word) in enumerate(words):
            wlen = len(word)
            for j in range(wlen-1):
                av_trans[i] += transitions[w_i+j+1,w_i+j]
                av_trans[i+len(words)] += transitions[w_i+wlen-j-2,
                                                      w_i+wlen-j-1]
            w_i += wlen
            av_trans[i] /= wlen-1
            av_trans[i+len(words)] /= wlen-1
        figure()
        bar(arange(shape(av_trans)[0]),av_trans,\
            align='center')
        ax = gca()
        ax.set_xticks(arange(len(words)*2))
        ax.set_xticklabels(words + [x[::-1] for x in words],rotation=30)
        #~ title('Pattern Frequencies')
        ylabel('Average transition probability')
        tight_layout()
        utils.saveplot('AvTrans_%s.%s'\
                        %(data.c.stats.file_suffix[0],ftype))
    
### Plot SpontPCA
    #assumption: if spontpattern, then also spikes and indices
    if (data.__contains__('Spikes') and 
        data.__contains__('SpontPattern')):
        print 'plot SpontPCA'
        steps_noplastic_test = data.c.steps_noplastic_test[0]
        input_spikes = data.Spikes[0][:,0:-steps_noplastic_test]
        spont_spikes = data.Spikes[0][:,-steps_noplastic_test:]
        input_index = data.InputIndex[0][0:-steps_noplastic_test] 
        
        filename = os.path.join(pickle_dir,"source_train.pickle")
        source = pickle.load(gzip.open(filename,"r"))
        if isinstance(source,TrialSource):
            source = source.source
        max_word_length = max([len(x) for x in source.words])
        N_words = len(source.words)
        words = source.words
        words_subscript = [letter for word in words for letter in word]
        
        if data.c.stats.__contains__('only_last'):
            N_comparison = data.c.stats.only_last[0]\
                            -data.c.steps_noplastic_test[0]
        else:
            N_comparison = 2500
        assert(N_comparison > 0)
        assert(N_comparison <= steps_noplastic_test)
        maxindex = int(max(input_index))
        
        # Only use spikes that occured at the end of learning and spont
        last_input_spikes = input_spikes[:,-N_comparison:]
        last_spont_spikes = spont_spikes[:,-N_comparison:]
        last_input_index = input_index[-N_comparison:]
        
        # Filter out empty states
        last_input_spikes = last_input_spikes[:,last_input_index != -1]
        last_input_index = last_input_index[last_input_index != -1]
        
        # normalize data - this is a suggestion from Jochen
        if normalize_PCA:
            eps = 0.00000000000000000001
            summed_last_input_spikes = sum(last_input_spikes,0)
            last_input_spikes /= summed_last_input_spikes+eps
            summed_last_spont_spikes = sum(last_spont_spikes,0)
            last_spont_spikes /= summed_last_spont_spikes+eps

        # Apply PCA
        (input_pcaed, pcas, var) = pca(last_input_spikes.T)
        
        spont_pcaed = np.dot(last_spont_spikes.T,pcas)
        if not pca_animation: # causes the video to be garbage
            savefig_bbox = mpl.rcParams['savefig.bbox']
            mpl.rcParams['savefig.bbox'] = 'tight'
        fig = plt.figure()
        ax = fig.add_subplot(111,projection='3d')
        
        if not pca_animation:
            inputs = N_comparison
            sponts = 100
        else:
            inputs = 20 # To get limits roughly right
            sponts = 0
        
        cmap = plt.cm.Set1 # was jet
        cmaplist = [cmap(i) for i in range(cmap.N)]
        cmap = cmap.from_list('Custom cmap', cmaplist, cmap.N)
        bounds= arange(maxindex+2)
        norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
        ax2 = fig.add_axes([0.9, 0.1, 0.03, 0.8])
        pc_dim = [0,1,2]
        cb = mpl.colorbar.ColorbarBase(ax2, cmap=cmap, norm=norm, 
                              spacing='proportional', ticks=bounds[:-1], 
                              boundaries=bounds, format='%1i')    
        if platform.system() == 'Windows':
            args = {'c':cmap(last_input_index[:inputs].astype('int')*(256/bounds[-2]),1)}
        else:
            args = {'c':last_input_index[:inputs].astype('int'),
                    'cmap':cmap,'norm':norm}
        scat = ax.scatter(input_pcaed[:inputs,pc_dim[0]],
                   input_pcaed[:inputs,pc_dim[1]],
                   zs=input_pcaed[:inputs,pc_dim[2]],s=60, **args)
        # dummy plot because 3dscatter not supported by legend
        ax.plot([],[],'o',label='Evoked')
        line, = ax.plot(spont_pcaed[:sponts,pc_dim[0]],
                spont_pcaed[:sponts,pc_dim[1]],
                zs=spont_pcaed[:sponts,pc_dim[2]],
                c='b',linewidth=1,label='Spont')
        #~ ax.set_title('Evoked (dots) vs. Spontaneous (lines)')
        ax.legend(loc='upper right')
        # ax2.set_ylabel('Input Index', size=12)
        # this assumes that spontpattern exists
        #~ ax2.set_yticklabels(array([x for x in words_subscript]))
        letters = array([letter for word in words for letter in word])
        n_letters = len(letters)
        cb.set_ticks(linspace(0.5,n_letters-0.5,n_letters))
        cb.set_ticklabels(letters)
        ax.set_xlabel('PC%d'%(pc_dim[0]+1))
        ax.set_ylabel('PC%d'%(pc_dim[1]+1))
        ax.set_zlabel('PC%d'%(pc_dim[2]+1))
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])
        utils.saveplot('SpontPCA_%s.%s'\
                        %(data.c.stats.file_suffix[0],ftype))
                        
                        
        if pca_animation:
            interval = 100 # in ms
            N_frames = 50
            fps = 2
            from matplotlib import animation
            from mpl_toolkits.mplot3d.art3d import juggle_axes
            def init_evoked():
                ax.view_init(20,100)
                return scat,
            def animate_evoked(s):
                x = input_pcaed[0:s,pc_dim[0]]
                y = input_pcaed[0:s,pc_dim[1]]
                z = input_pcaed[0:s,pc_dim[2]]
                scat._offsets3d = (np.ma.ravel(x), np.ma.ravel(y), 
                                   np.ma.ravel(z))
                scat._facecolor3d = scat.to_rgba(
                                    last_input_index[0:s].astype('int'))
                #~ ax.view_init(20, 100 + 0.3 * s)
                draw()
                return scat,
            
            anim = animation.FuncAnimation(fig,animate_evoked,
                                           init_func=init_evoked,
                                           interval=interval,
                                           frames=N_frames)
            anim.save(utils.logfilename('pca_anim_evoked.mp4'),
                      writer='avconv', fps=fps)
            
            def init_spont():
                ax.view_init(20,100)
                return line,
            def animate_spont(s):
                x = spont_pcaed[0:s,pc_dim[0]]
                y = spont_pcaed[0:s,pc_dim[1]]
                z = spont_pcaed[0:s,pc_dim[2]]
                line.set_xdata(x)
                line.set_ydata(y)
                line.set_3d_properties(z)
                #~ ax.view_init(20, 100 + 0.3 * s)
                draw()
                return line,
            anim = animation.FuncAnimation(fig,animate_spont,
                                           init_func=init_spont,
                                           interval=interval,
                                           frames=N_frames)
            anim.save(utils.logfilename('pca_anim_spont.mp4'),
                      writer='avconv', fps=fps)
            
        if not pca_animation:                
            mpl.rcParams['savefig.bbox'] = savefig_bbox
        # For the rate plots
        if normalize_PCA:
            last_input_spikes *= summed_last_input_spikes
            last_spont_spikes *= summed_last_spont_spikes
        
        figure()
        plot(100*(cumsum(var)/sum(var))[:20])
        ylabel('Variance explained [%]')
        xlabel('Principal component')
        utils.saveplot('SpontPCA_Variance_%s.%s'\
                        %(data.c.stats.file_suffix[0],ftype))
                        
        figure()
        
        rates = np.zeros(maxindex+1)
        for i in range(maxindex+1):
            rates[i] = mean(sum(last_input_spikes[:,last_input_index==i]
                                ,0))/data.c.N_e[0]
        
        bar(arange(maxindex+1),rates,align='center')
        xticks(arange(maxindex+1),array([x for x in words_subscript]))
        ylabel('Mean rate')
        utils.saveplot('SpontPCA_Rates_%s.%s'\
                        %(data.c.stats.file_suffix[0],ftype))
    
                   
    ### Plot Bayes
    if data.__contains__('Bayes'):
        print 'plot Bayes'   
        frac_A = data.c.frac_A[0]
        bayes = data.Bayes[0]
        
        fig, ax = plt.subplots()
        plot(frac_A*100,bayes[:,0],'-b',label='Decision A')
        hold('on')
        plot(frac_A*100,bayes[:,1],'-g',label='Decision B')
        ylim([0,1])
        xlabel('Percentage of A in ambiguous stimulus')
        ylabel('Output gain')
        tight_layout()                       
        utils.saveplot('bayes_drive_%s.%s'\
                        %(data.c.stats.file_suffix[0],ftype))
        
        if shape(bayes)[1]>4:
            fig, ax = plt.subplots()
            plot(frac_A*100,bayes[:,4],'-b',label='Decision A')
            hold('on')
            plot(frac_A*100,bayes[:,5],'-g',label='Decision B')
            ylim([0,1])
            legend(loc='upper center')
            xlabel('Percentage of A in ambiguous stimulus')
            ylabel('Fraction of decisions')
            tight_layout()
            
            from chartmann.spont.optimal_channels import OptimalChannels
            
            filename = os.path.join(pickle_dir,"source_train.pickle")
            source = pickle.load(gzip.open(filename,"r"))
            if isinstance(source,TrialSource):
                source = source.source
            p_A = source.probs[0,0]
            
            channels = OptimalChannels(N_u=data.c.N_u_e[0])
            N_As = (frac_A*data.c.N_u_e[0]).round().astype(int)
            
            def opt_wrapper(x,p_uA_given_A,p_uA_given_B):
                channels.p_uA_given_A = p_uA_given_A
                channels.p_uB_given_B = p_uA_given_A
                channels.p_uA_given_B = p_uA_given_B
                channels.p_uB_given_A = p_uA_given_B
                return channels.optimal_inference(p_A=p_A,N_As=x,
                                                  N_samples=1000)
            
            #~ ((p_uA_given_A,p_uA_given_B),pcov) = curve_fit(
                            #~ opt_wrapper,N_As,bayes[:,4],p0=[0.6,0.4])
            
            ps_uA_given_A = linspace(0,1,10)
            ps_uA_given_B = linspace(0,1,10)
            best_so_far = inf
            y = bayes[:,4]
            for pAA in ps_uA_given_A:
                for pAB in ps_uA_given_B:
                    if pAA>pAB: # Can be swapped w/ same result
                        y_est = opt_wrapper(N_As,pAA,pAB)
                        dist = np.linalg.norm(y-y_est)
                        if dist<best_so_far:
                            p_uA_given_A = pAA
                            p_uA_given_B = pAB
                            best_so_far = dist
            
            fitted_channels = OptimalChannels(p_uA_given_A=p_uA_given_A,
                                       p_uA_given_B=p_uA_given_B,
                                       N_u=data.c.N_u_e[0])
            opt_posteriors = fitted_channels.optimal_inference(p_A=p_A,
                                               N_As=N_As,N_samples=1000)
            plot(frac_A*100,opt_posteriors,'--b',label='Prob. model A')
            plot(frac_A*100,1-opt_posteriors,'--g',label='Prob. Model B')
            title('p(success|input)=%.1f   p(sucess|noinput)=%.1f'%(
                                             p_uA_given_A,p_uA_given_B))
            legend(loc='best')

            # Optimality based on actual activation probabilities
            # (not equal to channel success/failure)
            if 'AA_probs' in locals() and False:
                filename = os.path.join(pickle_dir,
                                                  "source_train.pickle")
                source = pickle.load(gzip.open(filename,"r"))
                if isinstance(source,TrialSource):
                    source = source.source
                p_A = source.probs[0,0]
                # Sanitize
                eps = 0.01 
                AA_probs[AA_probs<eps] += eps
                AB_probs[AB_probs<eps] += eps
                BA_probs[BA_probs<eps] += eps
                BB_probs[BB_probs<eps] += eps
                AA_probs[(1-AA_probs)<eps] -= eps
                AB_probs[(1-AB_probs)<eps] -= eps
                BA_probs[(1-BA_probs)<eps] -= eps
                BB_probs[(1-BB_probs)<eps] -= eps
                
                # Control to compare to simpler model
                #~ AA_probs = ones(shape(AA_probs))*0.5
                #~ AB_probs = ones(shape(AB_probs))*0.3
                #~ BA_probs = ones(shape(BA_probs))*0.3
                #~ BB_probs = ones(shape(BB_probs))*0.5
            
                # A_on is vector of length N_u with 1 if A-receiving unit was activated
                def p_A_given_AB_on(A_on,B_on):
                    # Bool necessary to index properly
                    assert(A_on.dtype == bool and B_on.dtype == bool)
                    A_off = ~A_on
                    assert(sum(A_on+A_off)==data.c.N_u_e[0] 
                            and A_off.dtype == bool)
                    B_off = ~B_on
                    p_B = 1-p_A
                    # Assuming independence
                    p_AB_on_given_A = (AA_probs[A_on].prod() 
                                        * (1-AA_probs)[A_off].prod() 
                                        * BA_probs[B_on].prod() 
                                        * (1-BA_probs)[B_off].prod())
                    p_AB_on_given_B = (AB_probs[A_on].prod() 
                                        * (1-AB_probs)[A_off].prod() 
                                        * BB_probs[B_on].prod() 
                                        * (1-BB_probs)[B_off].prod())
                    return (p_AB_on_given_A*p_A)/((p_AB_on_given_A*p_A)
                                                 +(p_AB_on_given_B*p_B))
                    
                def sample_posteriors(fracs_A):
                    N_u = data.c.N_u_e[0]
                    N_samples = 100
                    N_values = shape(fracs_A)[0]
                    posteriors = zeros((N_values,N_samples))
                    
                    for (i,frac) in enumerate(fracs_A):
                        # The amb. stimuli are constructed as follows
                        # (for historical reasons ;-))
                        # A_units = [A1,A2,A3,A4,...,A10]
                        # B_units = [B1,B2,B3,B4,...,B10]
                        # 30%A70%B= [A1,A2,A3,B4,...,B10] 
                        # Therefore, we have to treat the FIRST 30% of 
                        # A units as activated by A and the rest as Bact
                        # And the LAST 70% of B units as activated by B
                        N_A = int(round(frac*N_u))
                        N_B = int(round((1-frac)*N_u))
                        assert(N_A+N_B==N_u)
                        A_given_A = ones(N_A)*AA_probs[:N_A]
                        A_given_B = ones(N_B)*AB_probs[N_A:]
                        A_on_template = hstack((A_given_A,A_given_B))
                        assert(len(A_on_template)==N_u)
                        B_given_A = ones(N_A)*BA_probs[:N_A]
                        B_given_B = ones(N_B)*BB_probs[N_A:]
                        B_on_template = hstack((B_given_A,B_given_B))
                        
                        for j in range(N_samples):
                            A_on = A_on_template>rand(N_u)
                            B_on = B_on_template>rand(N_u)
                            
                            posteriors[i,j] = p_A_given_AB_on(A_on,B_on)
                            
                    return posteriors
                
                posteriors = sample_posteriors(frac_A)
                plot(frac_A*100,mean(posteriors,1),'--',
                     label='Prob. model A')
                plot(frac_A*100,1-mean(posteriors,1),'--',
                     label='Prob. model B')
                legend(loc='best')

            utils.saveplot('bayes_dec_%s.%s'\
                            %(data.c.stats.file_suffix[0],ftype))

    if data.__contains__('SpontBayes'):
        print 'plot SpontBayes'
        sb = data.SpontBayes[0]
        # over all conditions: check if higher-than-mean readout
        # corresponds to higher-than-mean activation of input units
        mean_readout = mean(sb,1)[:,2:]
        mean_act = mean(sb,1)[:,:2]
        n_conditions = shape(sb)[0]
        relative_effect = np.zeros((n_conditions,2))
        for i in range(n_conditions):
            indices_0 = where(sb[i,:,2]>mean_readout[i,0])[0]
            indices_1 = where(sb[i,:,3]>mean_readout[i,1])[0]
            relative_effect[i,0] = mean(sb[i,indices_0,0])/mean_act[i,0]
            relative_effect[i,1] = mean(sb[i,indices_1,1])/mean_act[i,1]
        figure()
        boxplot(relative_effect.flatten()*100)
        ylabel('Effect of better-than-mean readout on active'
               'input units [%]')
        utils.saveplot('spontbayes_%s.%s'%\
                                    (data.c.stats.file_suffix[0],ftype))
        figure()
        meandiff_0 = mean_act[:,0]-frac_A*data.c.N_u_e[0]
        meandiff_1 = mean_act[:,1]-frac_A[::-1]*data.c.N_u_e[0]
        plot(frac_A,meandiff_0,'-b',label='A predicted')
        hold('on')
        plot(frac_A,meandiff_1,'-g',label='B predicted')
        legend(loc='upper center')
        plot([0,1],[0,0],'--')
        xlabel('% A')
        ylabel('Average difference from input activity')
        title('Mean M: %.2f - Mean N: %.2f'%(mean(meandiff_0), \
                                                      mean(meandiff_1)))
        tight_layout()
        utils.saveplot('spondbayes_%s.%s'%\
                                    (data.c.stats.file_suffix[0],ftype))
        
    if data.__contains__('CondProb') and data.__contains__('endweight'):
        print 'plot Condprob'
        figure()
        x = data.endweight[0].flatten()
        y = data.CondProb[0].flatten()
        # Remove correlations for zero weight, because network cannot
        # insert synapses -> pointless to look at corresponding probs.
        y = y[x>0]
        x = x[x>0]
        scatter(x,y,s=60)
        gca().locator_params(axis='y',nbins=4) # 4 ticks/axis
        gca().locator_params(axis='x',nbins=4) # 4 ticks/axis
        xlim([0,1])
        ylim([0,1])
        #~ ylim([-0.1,1.1])
        #~ xlim([-0.1,1.1])       
        hold('on')
        #~ plot([0,1],[data.c.h_ip[0],data.c.h_ip[0]],'--')
        import scipy.stats as stats
        (r,p) = stats.pearsonr(x,y)
        #title('- all exc. units - (PCC=%.2f,p=%.2f)'%(r,p))
        xlabel('Synaptic strength')
        ylabel('Conditional probability of spiking')
        tight_layout()
        utils.saveplot('condprob_%s.%s'\
                        %(data.c.stats.file_suffix[0],ftype))
        
        if False:
            figure()
            noinput_units = arange(N_e)[data.InputUnits[0]==0]
            x = data.endweight[0][noinput_units,noinput_units].flatten()
            y = data.CondProb[0][noinput_units,noinput_units].flatten()
            #~ z = data.Rate[0][50:,50:].flatten() # no influence
            y = y[x>0]
            x = x[x>0]
            #~ z = z[x>0]
            scatter(x,y)
            #~ ylim([-0.1,1.1])
            #~ xlim([-0.1,1.1])
            hold('on')
            #~ plot([0,1],[data.c.h_ip[0],data.c.h_ip[0]],'--')
            import scipy.stats as stats
            (r,p) = stats.pearsonr(x,y)
            #title('- only reservoir - (PCC=%.2f,p=%.2f)'%(r,p))
            xlabel('Synaptic strength')
            ylabel('Conditional probability of spiking')
            tight_layout()
            utils.saveplot('condprob_reservoir_%s.%s'\
                            %(data.c.stats.file_suffix[0],ftype))    
    
    if data.__contains__('Balanced'):
        print 'plot Balanced'
        N_e = data.c.N_e[0]
        #~ insteps = data.c.steps_plastic[0]
        insteps = shape(data.Balanced[0])[1]
        inh_drive = data.Balanced[0][N_e:2*N_e,:insteps]
        exc_drive = data.Balanced[0][:N_e,:insteps]
        t_e_drive = data.Balanced[0][2*N_e:,:insteps]
        #~ #inh_drive += t_e_drive
        m_diff = mean(abs(mean(exc_drive,0)-mean(inh_drive,0)))
        steps = 200 # for plotting - last steps to plot
        
        # First plot means
        figure()
        plot(mean(exc_drive[:,-steps:],0),'b',label='Excitatory')
        hold('on')
        plot(mean(inh_drive[:,-steps:],0),'r',label='Inhibitory')
        #~ title('Mean Balance of Excitation and Inhibition')
        # Correlations:
        corr = pearsonr(mean(exc_drive[:,:],0),mean(inh_drive[:,
                        :],0))[0]
        #title('Pearson correlation: %.2f'%corr)
        xlabel('Step')
        ylabel('Mean drive ($\\rho_{E,I}=%.2f)$'%corr)
        legend(loc=4)
        tight_layout()
        utils.saveplot('meanbalance_%s.%s'\
                        %(data.c.stats.file_suffix[0],ftype)) 
        
        n_index = N_e-1# take a non-input one #randint(N_e)
        figure()
        scatter(exc_drive[n_index,-1000:],inh_drive[n_index,-1000:])
        correlations = np.zeros(N_e)
        for i in range(N_e):
            correlations[i] = pearsonr(exc_drive[i,-1000:],
                                       inh_drive[i,-1000:])[0]
        
        
        # Then plot one example neuron
        figure()
        n_index = N_e-1 # take a non-input one #randint(N_e)
        plot(exc_drive[n_index,-steps:],'b',label='Excitatory')
        hold('on')
        plot(t_e_drive[n_index,-steps:],'k--',label='$T_e$')
        plot(inh_drive[n_index,-steps:]+t_e_drive[n_index,-steps:],'r',
             label='Inhibitory+$T_e$')

        #~ title('Balance of Excitation and Inhibition (neuron:%d)'
                 #~ %n_index)
        xlabel('Step')
        ylabel('Drive (single neuron)')
        legend(loc=4)
        tight_layout()
        
        utils.saveplot('balance_%s.%s'\
                        %(data.c.stats.file_suffix[0],ftype))
                        
    # quenching variability
    if data.__contains__('Spikes') and\
        data.c.stats.quenching is not None:
            
        print 'plot quenching'
        
        mode = data.c.stats.quenching[0]
        assert(mode == 'train' or mode == 'test')

        spikes_before = 10
        spikes_after = 10
        # number of bins left and right of t (2 -> boxwidth=5)
        window_width = data.c.stats.quenching_window[0]
        weighted_regression = True
        matlab_comparison = use_matlab
        if platform.system() is 'Windows':
            matlab_comparison = False
        
        filename = os.path.join(pickle_dir,"source_%s.pickle"%mode)
        source = pickle.load(gzip.open(filename,"r"))
        if isinstance(source,TrialSource):
            source = source.source
        word_lengths = [len(x) for x in source.words]
        words = source.words
        max_word_length = max(word_lengths)
        N_words = len(source.words)
        total_length = max_word_length + spikes_before + spikes_after
        
        steps_plastic = data.c.steps_plastic[0]
        steps_noplastic_train = data.c.steps_noplastic_train[0]
        steps_noplastic_test = data.c.steps_noplastic_test[0]
        
        if mode == 'train':
            interval = [-steps_noplastic_train-steps_noplastic_test,
                        -steps_noplastic_test]
        else: # test because of assert
            interval = [steps_plastic+steps_noplastic_train,-1]
        
        input_spikes = data.Spikes[0][:,interval[0]:interval[1]]
        input_index = data.InputIndex[0][interval[0]:interval[1]]
        
        # build trial matrix (condition x trial x t x spikes)
        min_trials = inf
        word_start = 0
        for i in range(N_words):
            indices = find(input_index==word_start)
            tmp_trials = sum((indices >= spikes_before) * \
                        (indices <= shape(input_index)[0]-spikes_after))
            if tmp_trials < min_trials:
                min_trials = tmp_trials
            word_start += word_lengths[i]
                
        N_e = shape(input_spikes)[0]
        trials = np.zeros((N_words,min_trials,total_length,N_e))
        word_start = 0
        for word in range(N_words):
            indices = find(input_index==word_start)
            indices = indices[((indices >= spikes_before) * \
                         (indices <= shape(input_index)[0]- \
                         (spikes_after+max_word_length)))]
            indices = indices[-min_trials:]
            for (trial,i) in enumerate(indices):
                trials[word,trial,:,:] = input_spikes[:,i-spikes_before:
                                       i+max_word_length+spikes_after].T
            word_start += word_lengths[word]

        x = linspace(-spikes_before+2*window_width, 
                     spikes_after+max_word_length-1,
                     total_length-(window_width*2))
        
        noinput_units = arange(N_e)[data.InputUnits[0]==0]
        
        FF = np.zeros((N_words,total_length-2*window_width))
        means = np.zeros((N_words,total_length-2*window_width))
        allvars = np.zeros((N_words,total_length-2*window_width))
        import scipy.stats as stats
        if matlab_comparison:
            from mlabwrap import mlab
            mlab.addpath(
                     '/home/chartmann/Desktop/sorn/py/chartmann/spont/')
            #~ result = mlab.VarVsMean_pythontomat(trials)
            #~ N = N_words
            result = mlab.VarVsMean_pythontomat_bulk(trials[:,:,:,
                                                         noinput_units])
            N = 1
            FFs_mlab = result[:,:N].T
            # this is minmax minmax minmax
            FFs_mlab_95CIs = result[:,N:3*N].T 
            means_mlab = result[:,3*N:4*N].T/1000.
            FFsAll_mlab = result[:,4*N:5*N].T
            # this is minmax minmax minmax
            FFsAll_mlab_95CIs = result[:,N*5:7*N].T 
            meansAll_mlab = result[:,7*N:8*N].T/1000.
            x_mlab = x[:shape(FFs_mlab)[1]]
            
            def remove_axes(ax):
                ax.spines["right"].set_visible(False)
                ax.spines["top"].set_visible(False)
                ax.spines["bottom"].set_visible(False)
                ax.tick_params(axis='both', direction='out')
                ax.get_yaxis().tick_left()
                ax.get_xaxis().set_visible(False)
            
            
            import matplotlib.gridspec as gridspec

            lw = 3
            figure()
            gs = gridspec.GridSpec(2, 1,height_ratios=[0.4,0.6])
            subplot(gs[0])
            #~ ax = axes(frameon=False)
            #~ ax.get_xaxis().set_visible(False)
            plot(x_mlab,meansAll_mlab[0],'k',label="Raw",lw=lw)
            plot(x_mlab,means_mlab[0],c='0.5',label="'Matched'",lw=lw)
            #~ gca().locator_params(axis='y',nbins=2) # 4 ticks/axis
            minmax = [min(hstack((meansAll_mlab[0],means_mlab[0]))), 
                      max(hstack((meansAll_mlab[0],means_mlab[0])))]
            minmax[0] = round(minmax[0]-0.0049,3)
            minmax[1] = round(minmax[1]+0.0049,3)
            minmaxx = [x_mlab[0]-1,max(x_mlab)]
            ylabel('Spikes/step')
            gca().locator_params(axis='y',nbins=4)
            remove_axes(gca())
            legend(loc='best')
            ylim(minmax)
            xlim(minmaxx)
            tight_layout()
            subplot(gs[1])
            plot(x_mlab,FFs_mlab[0],'k',label='FF',lw=lw)
            plot(x_mlab,FFs_mlab_95CIs[0],c='0.5',label='95CI',lw=lw)
            plot(x_mlab,FFs_mlab_95CIs[1],c='0.5',lw=lw)
            quiver(-3,ylim()[0],0,0.1,scale=1,label='Stim on')
            gca().locator_params(axis='y',nbins=4)
            remove_axes(gca())
            legend(loc='best')
            ylabel('Fano Factor')
            xlim(minmaxx)
            tight_layout()
            utils.saveplot('quenching_matlab_%s.%s'\
                                %(data.c.stats.file_suffix[0],ftype))

        
        for word in range(N_words):
            for (i,t) in enumerate(arange(0,
                                          total_length-2*window_width)):
                # Take this procedure from quenching variability paper
                # figure 4:
                # Regress between means and variances for all neurons in
                # small interval (in our case in single step) over
                # trials
                # Selecting neurons by fancy indexing (noinput_units is 
                # array), yields a restructuring to neurons x trials x t
                # due to advanced indexing
                # see http://stackoverflow.com/questions/11942747/numpy-multi-dimensional-array-indexing-swaps-axis-order
                # http://docs.scipy.org/doc/numpy/reference/ufuncs.html#arrays-broadcasting-broadcastable
                # Basically, word is broadcasted to 
                # [word]*len(noinput_units)
                # It is then ambiguous where to put this subspace 
                # -> put to front
                # 1. take spike count in window
                count = sum(trials[word,:,t
                            :t+2*window_width+1,noinput_units],2)
                # 2a. mean of spike count over all trials
                meanss = mean(count,1)
                means[word,i] = mean(meanss)
                # 2b. var of spike count over all trials
                varss = std(count,1)**2 
                allvars[word,i] = mean(varss)
                # 3. regress over all neurons
                #~ (slope, intercept, r_value, p_value, std_err) = \
                        #~ stats.linregress(meanss,varss)
                #~ # more or less same thing:
                #~ (slope, offset) = np.polyfit(meanss,varss,1)
                weighting = eye(shape(meanss)[0])
                # see paper 
                # + http://www.math.uah.edu/stat/point/Estimators.html
                if weighted_regression:
                    for j in range(shape(meanss)[0]):
                        weighting[j,j] = min_trials/((meanss[j]+
                                                      0.001)**2) 
                # through origin
                slope = np.dot(np.dot(meanss.T,weighting),varss)/np.dot(
                                      meanss.T,np.dot(weighting,meanss)) 
                FF[word,i] = slope
                # Sanity check
                #~ #figure()
                #~ #scatter(meanss,varss)
            # More sanity checks
            #~ if matlab_comparison:
                #~ figure()
                #~ plot(FFs_mlab[word],label='mlab')
                #~ hold('on')
                #~ plot(FF[word][3:],label='python')
                #~ legend()
            if word==0 or word==1:
                figure()
                subplot(2,1,1)
                axvspan(0,word_lengths[word]-1,color='#E6E6E6')
                hold('on')
                plot(x,FF[word],'k')                
                ylabel('Fano factor')
                locator_params(axis='y',nbins=4) # 4 ticks/axis
                title('%s'%words[word])
                #~ legend(loc='lower left')
                
                ax1 = subplot(2,1,2)
                ax1 = gca()
                axvspan(0,word_lengths[word]-1,color='#E6E6E6')
                plot(x,allvars[word],'b')
                locator_params(axis='y',nbins=4) # 4 ticks/axis
                ax1.yaxis.label.set_color('blue')
                ax2 = ax1.twinx()
                ax2.plot(x,means[word],'r')
                ax2.yaxis.label.set_color('red')
                ax2.set_ylabel('Mean rate')
                locator_params(axis='y',nbins=4) # 4 ticks/axis
                ax1.set_ylabel('Variance')
                ax2.set_xlabel('Step')
                #tight_layout()
                utils.saveplot('quenching_word_%d_%s.%s'\
                              %(word,data.c.stats.file_suffix[0],ftype))
                
                # plot sample spikes for random neuron                
                figure(figsize=(8,6)) # default (8,6)
                # [left, bottom, width, height]
                ax_scatter = plt.axes([0.1,0.32,0.85,0.65])
                ax_hist = plt.axes([0.1,0.12,0.85,0.2])
                from matplotlib.ticker import NullFormatter
                nullfmt   = NullFormatter()
                ax_scatter.xaxis.set_major_formatter(nullfmt)
                plt.sca(ax_scatter)
                N_trials = shape(trials)[1]
                y_lim_plot = [0,N_trials]
                axvspan(0,word_lengths[word]-1,color='#E6E6E6')
                x_lim = [-spikes_before,shape(trials)[2]-spikes_before]
                xlim(x_lim)
                neuron = randint(shape(trials)[3])
                for i in range(N_trials):
                    s_train = where(trials[word,i,:,neuron]==1)[0]\
                                                          -spikes_before
                    plot(s_train,np.ones(s_train.shape)*i,'o',color='k',
                          markersize=1)
                    hold('on')
                ylim(y_lim_plot)
                gca().set_yticks([])
                ylabel('Trial')
                plt.sca(ax_hist)
                axvspan(0,word_lengths[word]-1,color='#E6E6E6')
                plt.bar(arange(total_length)-spikes_before,
                        sum(trials[word,:,:,neuron],0)/N_trials,
                        align='center')
                xlim(x_lim)
                xlabel('Step')
                ylabel('Rate')
                gca().locator_params(axis='y',nbins=4) # 4 ticks/axis
                utils.saveplot('quenching_spikes_word_%d_%s.%s'
                            %(word,data.c.stats.file_suffix[0],ftype))
        
    if data.__contains__('TrialBayes'):
        print 'plot TrialBayes'
        figure()
        filename = os.path.join(pickle_dir,"source_test.pickle")
        source_test = pickle.load(gzip.open(filename,"r"))
        if isinstance(source_test,TrialSource):
            source_test = source_test.source
        word_lengths = [len(x) for x in source_test.words]
        word_length = max(word_lengths)
        ig = 30
        forward_pred = data.c.stats.forward_pred[0]
        x = arange(-ig+forward_pred,forward_pred)
        tmp = data.TrialBayes[0][0]*100
        nonzero_set = set(where(tmp!=-100)[0])
        nonzero_list = [n for n in nonzero_set]
        agreement = tmp[nonzero_list]
        agreement_lstsq = data.TrialBayes[0][1]*100
        agreement_lstsq = agreement_lstsq[nonzero_list]
        if len(nonzero_list)>0:
            plot(x,agreement[-ig:],color='#808080',label='Baseline')
            plot(x,agreement_lstsq[-ig:],color='b',
                 label='Spont. prediction')
            hold('on')
            y_lim = ylim()
            axvspan(0,word_length-1,color='#E6E6E6')
            plot([word_length-1,word_length-1],y_lim,'--g',
                 label='Pred. position')
            xlabel('Prediction step before input')
            ylabel('Pred-decision agreement [%]')        
            tight_layout()
            legend(loc="upper left")
            utils.saveplot('trialbayes_%s.%s'\
                        %(data.c.stats.file_suffix[0],ftype))
                    
    if data.__contains__('AttractorDynamics'):
        print 'plot AttractorDynamics'
        # This is now frac x step (from close to cue to close to target)
        output_dists = data.AttractorDynamics[0].T[:,::-1]
        frac_A = data.c.frac_A[0]
        x = arange(-shape(output_dists)[1]+1,1)
        figure()
        for (i,frac) in enumerate(frac_A):
            plot(x,output_dists[i,:],label="%.2f"%frac)
        ylabel('Distance between output gains')
        xlabel('Steps before target')
        legend()
        tight_layout()
        utils.saveplot('attractordynamics_%s.%s'\
                    %(data.c.stats.file_suffix[0],ftype))
                        
    if data.__contains__('OutputDist'):
        print 'plot OutputDist'
        output_dist = data.OutputDist[0][0]
        output_std  = data.OutputDist[0][1]
        frac_A = data.c.frac_A[0]
        
        figure()
        errorbar(frac_A*100, output_dist, output_std)
        ylim([0,1])
        xlabel('Percentage of A in ambiguous stimulus')
        ylabel('Output gain difference +- std')
        utils.saveplot('outputdist_%s.%s'\
                        %(data.c.stats.file_suffix[0],ftype))
                        
    if data.__contains__('EvokedPred'):
        print 'plot EvokedPred'
        # First plot average evoked pred over time
        filename = os.path.join(pickle_dir,"source_%s.pickle"
                                            %data.c.stats.quenching[0])
        source = pickle.load(gzip.open(filename,"r"))
        if isinstance(source,TrialSource):
            source = source.source
        word_lengths = [len(x) for x in source.words]
        word_length = max(word_lengths)

        figure()
        pred_spont = data.EvokedPred[0][:,:-word_length,0]
        pred_base  = data.EvokedPred[0][:,:-word_length,1]
        x = arange(shape(pred_spont)[1])

        errorbar(x,mean(pred_spont,0),std(pred_spont,0),color='b',\
                 label='Spont. pred.')
        hold('on')
        errorbar(x,mean(pred_base,0),std(pred_base,0),color='#808080',
                 label='Baseline')
        y_lim = ylim()
        axvspan(0,word_length-1,color='#E6E6E6')
        ylim(y_lim)
        ax = gca()
        # Reorder labels b/c ugly
        #~ handles, labels = ax.get_legend_handles_labels()
        #~ labels = [x for x in array(labels)[[2,3,0,1]]]
        #~ handles = [x for x in array(handles)[[2,3,0,1]]]
        #~ ax.legend(handles, labels)
        legend()
        xlabel('Step after stimulus onset')
        ylabel('Pearson Correlation +- std')
        title('Correlation between predicted and evoked activity')
        tight_layout()
        utils.saveplot('evokedpred_%s.%s'\
                        %(data.c.stats.file_suffix[0],ftype))
        
        if 'FF' in locals() and False:
            # Then plot evoked pred vs. FF (high FF should -> better ep)
            figure()
            diff = pred_spont[:,1]-pred_base[:,1]
            FFs = FF[:,11]
            scatter(FFs,diff)
            (s,p) = stats.pearsonr(FFs,diff)
            xlabel('Fano factor after stimulus onset')
            ylabel('Spontpred - Baseline')
            title('p = %.4f'%p)
            # Do linear regression fit
            A = vstack((FFs,ones(shape(FFs)[0]))).T
            w = pinv(A).dot(diff)
            y = FFs*w[0]+w[1]
            tmp = zip(FFs,y)
            tmp.sort()
            tmp = array(tmp)
            hold('on')
            plot(tmp.T[0],tmp.T[1])
            tight_layout()
            utils.saveplot('evokedpred_FF_%s.%s'\
                            %(data.c.stats.file_suffix[0],ftype))
        
        # Finally plot EP vs. condition at step 1 (indisting. at step0)
        # look at method for how predictions are sorted
        # --> almost sorted, but not averaged etc.

        if False: # This only works for Bayes because of the meaned-hack
            frac_A = data.c.frac_A[0]
            pred_p = pred_pinv[:,1]
            to_mean = pred_p[2:]
            meaned = [mean([x,y]) for (x,y) in zip(to_mean[::2],\
                                                   to_mean[1::2])]
            # B, C, D, ..., A
            pred_p = hstack((pred_p[1],array(meaned),pred_p[0]))
            pred_s = pred_base[:,1]
            to_mean = pred_s[2:]
            meaned = [mean([x,y]) for (x,y) in zip(to_mean[::2],\
                                                   to_mean[1::2])]
            pred_s = hstack((pred_s[1],array(meaned),pred_s[0]))
            figure()
            plot(frac_A*100,pred_p,label='Spont. pred.')
            hold('on')
            plot(frac_A*100,pred_s,label='Baseline')
            xlabel('Percentage of A in ambiguous stimulus')
            ylabel('Prediction')
            legend()
            tight_layout()
            utils.saveplot('evokedpred_fracA_%s.%s'\
                            %(data.c.stats.file_suffix[0],ftype))
    
    if data.__contains__('SVD'):
        print 'plot SVD'
        figure()
        N_steps = data.c.steps_plastic[0]
        x = linspace(0,N_steps,shape(data.SVD[0])[0])
        plot(x,data.SVD[0],'k-')
        hold('on')
        plot(x,mean(data.SVD[0],1),linewidth=3)
        xlabel('Step')
        ylabel('Singular values')
        utils.saveplot('SVD_values_%s.%s'\
                        %(data.c.stats.file_suffix[0],ftype))
        # Compute Transitions
        filename = os.path.join(pickle_dir,"source_plastic.pickle")
        source_plastic = pickle.load(gzip.open(filename,"r"))
        if isinstance(source_plastic,TrialSource):
            source_plastic = source_plastic.source
            
        steps_noplastic_test = data.c.steps_noplastic_test[0]
            
        words = source_plastic.words
        words_subscript = ["$\mathrm{%s}_{%i%i}$"%(
                            ("\%s" if letter == "_" else letter),i,j) \
                            for (i,word) in enumerate(words) \
                            for (j,letter) in enumerate(word)]
        words_subscript = [letter for word in words for letter in word]

        N_e = data.c.N_e[0]
        steps_noplastic_test = data.c.steps_noplastic_test[0]
        input_index = data.InputIndex[0][0:-steps_noplastic_test]
        maxindex = int(max(input_index))
        svd_steps = shape(data.SVD[0])[0]
        transitions = np.zeros((svd_steps,maxindex+1,maxindex+1))
        for s in xrange(svd_steps):
            for i in range(N_e):
                transitions[s,int(data.SVD_U[0][s,i]),
                            int(data.SVD_V[0][s,i])] += data.SVD[0][s,i]
            for i in range(maxindex+1):
                summed_transitions = sum(transitions[s,:,i])
                if summed_transitions>0:
                    transitions[s,:,i] /= summed_transitions # normalize

        figure()
        im = imshow(transitions[-1],interpolation='none',vmin=0, vmax=1)
        xlabel('From')
        ylabel('To')
        ax = gca()
        ax.set_xticks(arange(len(words_subscript)))
        ax.set_xticklabels(array([x for x in words_subscript]))
        ax.set_yticks(arange(len(words_subscript)))
        ax.set_yticklabels(array([x for x in words_subscript]))
        colorbar(im, use_gridspec=True)
        tight_layout()
        utils.saveplot('SVD_transitions_%s.%s'\
                        %(data.c.stats.file_suffix[0],ftype))
                        
        if data.__contains__('SpontTransition'):
            spont_transitions = data.SpontTransition[0]
            # copy&paste from sponttransition-plot TODO function?
            for i in range(shape(spont_transitions)[0]):
                spont_transitions[:,i] /= sum(spont_transitions[:,i])
            (a,b,r,p,stderr) = stats.linregress(
                                spont_transitions.flatten(),
                                transitions[-1].flatten())
            figure()
            plot([0,1],[b,a+b],'g')
            plot(spont_transitions.flatten(),
                 transitions[-1].flatten(),'.')
            axis('equal')
            xlabel('Spontaneous transition')
            ylabel('SVD transition')
            utils.saveplot('SVD_vs_spont_transitions_%s.%s'\
                        %(data.c.stats.file_suffix[0],ftype))
            
        if False: # No video for now
            from matplotlib import animation
            fig = figure()
            im=imshow(np.zeros(shape(transitions[0])),
                      interpolation='none',vmin=0, vmax=1)
            xlabel('From')
            ylabel('To')
            ax = gca()
            ax.set_xticks(arange(len(words_subscript)))
            ax.set_xticklabels(array([x for x in words_subscript]))
            ax.set_yticks(arange(len(words_subscript)))
            ax.set_yticklabels(array([x for x in words_subscript]))
            tight_layout()
            def init():
                im.set_array(transitions[0])
                return[im]
            def animate(s):
                im.set_array(transitions[s])
                return [im]
            interval = 100 # in ms
            anim = animation.FuncAnimation(fig,animate,init_func=init,
                           frames=svd_steps,interval=interval,blit=True)
            anim.save(utils.logfilename('SVD_transition_animation.mpeg')
                                        , fps=1000//interval)
        
        figure()
        subplot(2,1,1)
        bar(arange(len(words_subscript)),sum(transitions[-1],0),
                                                         align='center')
        ylabel('Summed outgoing')
        tick_params(axis='x',labelbottom='off')
        subplot(2,1,2)
        bar(arange(len(words_subscript)),sum(transitions[-1],1),
                                                         align='center')
        ylabel('Summed incomming')
        ax = gca()
        ax.set_xticks(arange(len(words_subscript)))
        ax.set_xticklabels(array([x for x in words_subscript]))
        utils.saveplot('SVD_summed_%s.%s'\
                        %(data.c.stats.file_suffix[0],ftype))
                        
        # Word predictions:
        word_pred = np.zeros(len(words)*2)
        # Forward words:
        cumm_i = 0
        for (i,word) in enumerate(words):
            word_pred[i] = sum(transitions[-1,cumm_i,:])
            for j in range(len(word)-1):
                word_pred[i] *= transitions[-1,cumm_i+j+1,cumm_i+j]
            cumm_i += len(word)
        # Reverse words:
        cumm_i = maxindex
        N_words = len(words)
        for (i,word) in enumerate(words[::-1]):
            word_pred[2*N_words-1-i] = sum(transitions[-1,cumm_i,:])
            for j in range(len(word)-1):
                word_pred[2*N_words-1-i] *= transitions[-1,cumm_i-j-1,
                          cumm_i-j]
            cumm_i -= len(word)
        figure()
        bar(arange(len(word_pred)),word_pred,align='center')
        ax = gca()
        ax.set_xticks(arange(len(words)*2))
        ax.set_xticklabels(words + [x[::-1] for x in words],rotation=30)
        ylabel('SVD-Prediction')
        tight_layout()
        utils.saveplot('SVD_pred_%s.%s'\
                        %(data.c.stats.file_suffix[0],ftype))
        
            
if __name__=='__main__':        
    plot_results(path,datafile)
    show()
