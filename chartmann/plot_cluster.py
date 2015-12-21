from __future__ import division
from pylab import *
from scipy.optimize import curve_fit
from scipy import stats
import tables
import matplotlib.cm as mplcm
import matplotlib.colors as colors

import sys
sys.path.insert(0,"..")
import os
import utils
utils.backup(__file__)

import cPickle as pickle
import gzip
from common.sources import TrialSource

from utils.plotting import pretty_mpl_defaults

matlab_comparison = True # for FF

# Figure type (eps,pdf,png,...)
ftype = 'pdf'
# Data to plot
path = r'/home/chartmann/Desktop/Meeting Plots/2015-12-10_weightdepfailure_alignment/cluster_long_06_01_2015-12-09_17-06-48/common'
datafile = 'result.h5'
 
'''
Label significant differences in bar plots
Adapted from http://stackoverflow.com/questions/11517986/indicating-the-statistically-significant-difference-in-bar-graph

Parameters:
x1: x-value in plot to start bar from
x2: x-value in plot to end bar at
Y1: vector of datapoints corresponding to x1
Y2: vector of datapoints corresponding to x2
ax: axis to be plotted on
'''
def label_diff(x1,x2,Y1,Y2,ax):
    # Testing
    assert(len(Y1)==len(Y2))
    (t,pval) = stats.ttest_ind(Y1,Y2)
    if pval>0.05:
        return
    
    # If significant, draw bar
    N = len(Y1)
    x = mean([x1,x2])
    # Estimate how high to draw
    y = max(mean(Y1)+1.*std(Y1)/sqrt(N),mean(Y2)+1.*std(Y2)/sqrt(N))

    # Draw
    props = {'connectionstyle':'bar,fraction=0.15','arrowstyle':'-',
                'lw':2,'color':'k'}
    ax.annotate('*', xy=(x,1.05*y), zorder=10, ha='center')
    ax.annotate('', xy=(x1,y), xytext=(x2,y), arrowprops=props)
    
    # Extend figure height if bar out of range
    ylimit = ax.get_ylim()
    maxy = 1.1*y
    if ylimit[1] < maxy:
        ax.set_ylim((ylimit[0],maxy))

def errorspan(x,y,yerr,**kwargs):
    # , gets first item in list
    line, = plot(x,y,**kwargs)
    fill_between(x,y-yerr,y+yerr,alpha=0.5,facecolor=line.get_color())

# This contains LOTS of code duplication with plot_single...
def plot_results(result_path,result):
    pretty_mpl_defaults()
    final_path = os.path.join(result_path,result)
    print final_path
    h5 = tables.openFile(final_path,'r')
    data = h5.root
    if os.path.isdir(data.c.logfilepath[0]):
        pickle_dir = data.c.logfilepath[0]
    else:
        pickle_dir = result_path
    plots_path = os.path.join('..','plots')
    if not os.path.exists(plots_path):
        os.mkdir(plots_path)
    os.chdir(plots_path)
    # This ONLY works for the cluster when N_params == N_cores
    #N = shape(data.paramtracker[0])[0]
    #params = np.array([x/(10.0)+0.1 for x in range(10)])[:,None]
    params = data.paramtracker[0]
    N_params = shape(data.paramtracker)[1]
    N_iterations = shape(data.paramtracker)[0]
    param_name = data.c.cluster.vary_param[0]
    param_name_plot = param_name
    if param_name == 'source.prob':
        filename = os.path.join(pickle_dir,
                                "source_plastic_%s_%.3f.pickle"
                                %(param_name,params[0]))
        source_plastic = pickle.load(gzip.open(filename,"r"))
        if isinstance(source_plastic,TrialSource):
            source_plastic = source_plastic.source
        words = source_plastic.words
        param_name_plot = 'prior(%s)'%words[0]
    elif param_name == 'W_ee.p_failure':
        param_name_plot = 'Failure probability'
    elif param_name == 'W_ee.bias':
        param_name_plot = 'Pot. bias'
        
    param_name_u = param_name.replace(' ','_')
    
    print 'Iterations:', N_iterations
    
    ### Plot Activity Stats
    if data.__contains__('activity') and False:
        N = N_params
        activities = np.zeros(N)
        lookback = 3000
        for i in range(N):
            for j in range(np.shape(data.activity)[0]):
                activities[i] += sum(data.activity[j][i][-lookback:])\
                                    /(lookback*1.0)
            activities[i] /= 1.0*np.shape(data.activity)[0]
        figure()
        plot(params,activities,'o')
        title('Average activity vs. %s (%s)'
              %(data.c.cluster.vary_param[0],
              data.c.stats.file_suffix[0]))
        xlabel('%s'%(data.c.cluster.vary_param[0]))
        ylabel('Activity')
        utils.saveplot('Activity_%s.%s'
                       %(data.c.stats.file_suffix[0],ftype))
        
    if data.__contains__('meanactivity'):  
        test_words = data.c.source.test_words[0]
        baseline = data.meanactivity[:,:,0]
        act = {}
        act_2nd = {}
        start = 1
        for word in test_words:
            length = len(word)
            act[word] = mean(data.meanactivity[:,:,start:start+length],
                             2)
            act_2nd[word] = data.meanactivity[:,:,start+1]
            start += length
        # Colors from figures from paper
        c_gray = '#929496'
        c_blue = '#33348e'
        c_red  = '#cc2229'
        c_green= '#33a457'
        # Colors from figures from paper
        ekw = dict(elinewidth=5,ecolor='k')#,capsize=0)
        col = {'ABCD':c_blue,'DCBA':c_red,'A_CD':c_red,'E_CD':c_green}
        
        if data.c.source.control:
            condition = 'Control'
        else:
            condition = 'Experimental'
            
        if data.c.cluster.vary_param[0] == 'source.control' \
           and 'DCBA' in test_words:
            figure()
            bar(1,mean(baseline,0)[0],
                yerr=std(baseline,0)[0]/sqrt(N_iterations),color=c_gray,
                error_kw=ekw,label='Baseline',align='center')
            bar(2,mean(act['ABCD'],0)[0],
                yerr=std(act['ABCD'],0)[0]/sqrt(N_iterations),
                color=c_blue,error_kw=ekw,label='ABCD',align='center')
            bar(3,mean(act['DCBA'],0)[0],
                yerr=std(act['DCBA'],0)[0]/sqrt(N_iterations),
                color=c_red,error_kw=ekw,label='DCBA',align='center')
            bar(5,mean(baseline,0)[1],
                yerr=std(baseline,0)[1]/sqrt(N_iterations),color=c_gray,
                error_kw=ekw,align='center')
            bar(6,mean(act['ABCD'],0)[1],
                yerr=std(act['ABCD'],0)[1]/sqrt(N_iterations),
                color=c_blue,error_kw=ekw,align='center')
            bar(7,mean(act['DCBA'],0)[1],
                yerr=std(act['DCBA'],0)[1]/sqrt(N_iterations),
                color=c_red,error_kw=ekw,align='center')
            tick_params(axis='x',which='both',bottom='off',top='off')
            # Test significances
            label_diff(1,2,baseline[:,0],act['ABCD'][:,0],gca())
            label_diff(2,3,act['ABCD'][:,0],act['DCBA'][:,0],gca())
            label_diff(5,6,baseline[:,1],act['ABCD'][:,1],gca())
            label_diff(6,7,act['ABCD'][:,1],act['DCBA'][:,1],gca())
            
            xlim([0,8])
            xticks([2,6],['Experiment','Control'])
            ylabel('Sequence magnitude')
            legend(loc='lower left')
            utils.saveplot('Mean_reverse_%s.%s'
                            %(data.c.stats.file_suffix[0],ftype))
                                              
        figure()
        errorbar(params,mean(act['ABCD'],0),yerr=std(act['ABCD'],0)
                                              /sqrt(N_iterations),c='k')
        xlabel(param_name_plot)
        ylabel('Magnitude')
        pdiff = (params[-1]-params[0])/10.
        xlim([params[0]-pdiff,params[-1]+pdiff])
        utils.saveplot('Mean_vs_%s_%s.%s'
                      %(param_name_u,data.c.stats.file_suffix[0],ftype))
        
        for (p,param) in enumerate(params):
            figure()
            start = 1
            for word in test_words:
                length = len(word)
                x = arange(1,length+1)
                errorbar(x,mean(data.meanactivity[:,p,start:start
                         +length],0), yerr=std(data.meanactivity[:,p,
                         start:start+length],0)/sqrt(N_iterations),
                         c=col[word],label=word)
                start += length
            xlabel('Letter')
            ylabel('Magnitude')
            legend(loc='best')
            xlim([0,length+1])
            title(param_name_plot+' = %.2f'%param)
            utils.saveplot('Mean_time_%s_%s_%.2f.%s'%\
               (data.c.stats.file_suffix[0],param_name_u,param,ftype))
               
            figure()
            bar(1,mean(baseline,0)[p],
                yerr=std(baseline,0)[p]/sqrt(N_iterations),color=c_gray,
                error_kw=ekw,label='Baseline',align='center')
            for (i,word) in enumerate(test_words):
                bar(i+2,mean(act[word],0)[p],
                    yerr=std(act[word],0)[p]/sqrt(N_iterations),
                    color=col[word],error_kw=ekw,label=word,
                    align='center')
            tick_params(axis='x',which='both',bottom='off',top='off',
                        labelbottom='off')
            xlim([0.5,i+2.5])
            xlabel(param_name_plot+' = %.2f'%param)
            ylabel('Sequence magnitude')
            legend(loc='upper left')
            title(param_name_plot+' = %.2f'%param)
            utils.saveplot('Mean_reverse_%s_%s_%.2f.%s'%\
               (data.c.stats.file_suffix[0],param_name_u,param,ftype))
            
            figure()
            for (i,word) in enumerate(test_words):
                bar(i+1,mean(act[word],0)[p],
                    yerr=std(act[word],0)[p]/sqrt(N_iterations),
                    color=col[word],error_kw=ekw,align='center',
                    label=word)
                # Test significance
                for (j,word_cp) in enumerate(test_words[i+1:]):
                    label_diff(i+1,j+i+2,act[word][:,p],
                               act[word_cp][:,p],gca())
            l = i+1
            for (i,word) in enumerate(test_words):
                bar(i+2+l,mean(act_2nd[word],0)[p],
                    yerr=std(act_2nd[word],0)[p]/sqrt(N_iterations),
                    color=col[word],error_kw=ekw,align='center')
                # Test significance
                for (j,word_cp) in enumerate(test_words[i+1:]):
                    label_diff(i+2+l,j+i+3+l,act_2nd[word][:,p],
                               act_2nd[word_cp][:,p],gca())
            legend(loc='lower left')
            tick_params(axis='x',which='both',bottom='off',top='off')
            xticks([i//2+1,l+3],['Full sequence','Second element'])
            xlim([0,2*(i+1)+2])
            ylabel('Magnitude')
            #~ title(param_name_plot+' = %.2f'%param)
            utils.saveplot('Mean_2nd_%s_%s_%.2f.%s'%\
               (data.c.stats.file_suffix[0],param_name_u,param,ftype))
               
    if (data.__contains__('meanpattern') 
        and data.__contains__('meanactivity')):
        test_words = data.c.source.test_words[0]
        pats = {}
        start = 1
        for word in test_words:
            length = len(word)
            pats[word] = data.meanpattern[:,:,start:start+length]
            start += length
            
        if ('ABCD' in test_words and 'A_CD' in test_words and 'E_CD' in 
            test_words):
            for (p,param) in enumerate(params):
                figure()
                dist_con = sum(abs(pats['E_CD'][:,p,1,None]
                                    -pats['ABCD'][:,p,:]),2)
                dist_exp = sum(abs(pats['A_CD'][:,p,1,None]
                                    -pats['ABCD'][:,p,:]),2)
                bar(1,mean(dist_con[:,1]),
                    yerr=std(dist_con[:,1])/sqrt(N_iterations),
                    color=col['E_CD'],error_kw=ekw,align='center')
                bar(2,mean(dist_exp[:,1]),
                    yerr=std(dist_exp[:,1])/sqrt(N_iterations),
                    color=col['A_CD'],error_kw=ekw,align='center')
                label_diff(1,2,dist_con[:,1],dist_exp[:,1],gca())
                xticks([1,2],['E_CD','A_CD'])
                y_lim = ylim()
                ylim([0,y_lim[1]*1.1])
                ylabel('Manhattan distance')
                utils.saveplot('Mean_dist_%s_%s_%.2f.%s'%
                   (data.c.stats.file_suffix[0],param_name_u,param,
                   ftype))

    ### Plot endweight Stat
    if False and data.__contains__('endweight'):
        # First the logweight:
        logweight = data.endweight[0][data.endweight[0]>0]
        figure()
        logbins = logspace(-2,0,10)
        (y,_) = histogram(logweight,bins=logbins)
        #fit data to lognormal
        x = logbins[:-1]+(logbins[0]+logbins[1])/2.0
        semilogx(x,y,'.')

        # Do the fitting
        def lognormal(x,mue,var,scale):
            return scale * (exp(- ((log(x)-mue)*(log(x)-mue)) 
                    / (2*var)) / (x*sqrt(2*pi*var)))

        popt, pcov = curve_fit(lognormal, x, y)
        curve_x = logspace(-2,0,100)
        fitted_y = lognormal(curve_x,*popt)
        plot(curve_x,fitted_y)
        title('Final Weight Distribution (%s)'
                %(data.c.stats.file_suffix[0]))
        xlabel('Weight')
        ylabel('Frequency')
        legend(('data', 'lognormal fit (mue=%.3f var=%.3f scale=%.3f)'
                %(popt[0], popt[1], popt[2])))
        utils.saveplot('LogWeights_%s.%s'
                        %(data.c.stats.file_suffix[0],ftype))
        
        # Now scale-free property
        tmp = data.endweight[0]>0.0+0.0
        binary_connections = tmp+0.0
        in_degree = sum(binary_connections,1)
        out_degree = sum(binary_connections,0)
        fig = figure()
        fig.add_subplot(131)
        hist(in_degree)
        ylabel('frequency')
        xlabel('in degree')
        fig.add_subplot(132)
        hist(out_degree)
        xlabel('out degree')
        fig.add_subplot(133)
        hist(in_degree+out_degree)
        xlabel('in+out degree')
        plt.suptitle('Degree distributions')
        utils.saveplot('Degree_Distributions_%s.%s'
                       %(data.c.stats.file_suffix[0],ftype))
    
                   
    if False and (data.__contains__('Spikes') and data.__contains__('endweight')
        and data.__contains__('Bayes')):
        steps_plastic = data.c.steps_plastic[0]
        steps_noplastic_train = data.c.steps_noplastic_train[0]
        steps_noplastic_test = data.c.steps_noplastic_test[0]
       # TODO Plot response probabilities of input units from plot_single
        
    if data.__contains__('smallworld'):
        figure()
        gamma = np.zeros(N)
        lam = np.zeros(N)
        S_W = np.zeros(N)
        print data.smallworld
        for (i,item) in enumerate(data.smallworld):
            gamma += item.T[0][:N]
            lam += item.T[1][:N]
            S_W += item.T[2][:N]
        gamma /= (1.0*shape(data.smallworld)[0])
        lam /= (1.0*shape(data.smallworld)[0])
        S_W /= (1.0*shape(data.smallworld)[0])
        for i in range(N):
            plot([1,2,3],[gamma[i],lam[i],S_W[i]],'o')
        plot([0,4],[1,1],'--')
        legend(params)
        xticks([1,2,3],['gamma','lambda','S_W'])
        title('Small-world-ness with respect to %s'
              %data.c.cluster.vary_param[0])
        utils.saveplot('small_world_%s.%s'
                       %(data.c.stats.file_suffix[0],ftype))

    ### Plot ISIs
    if False and data.__contains__('ISIs'):
        figure()
        x = np.array(range(0,50))
        plot(x,data.ISIs[0][:], '.')

        # Do the fitting
        def exponential(x, a, b):
            return a * np.exp(-b*x)
        popt, pcov = curve_fit(exponential, x, data.ISIs[0][:])
        fitted_y = exponential(x,*popt)
        plot(x,fitted_y)

        title('Interspike Intervals (%s)'%(data.c.stats.file_suffix[0]))
        xlabel('ISI (Time Step)')
        ylabel('Frequency')

        legend(('data', 'exp fit (scale:%.3f exponent:%.3f)'
                        %(popt[0],-popt[1])))
        utils.saveplot('ISIs_%s.%s'%(data.c.stats.file_suffix[0],ftype))

    ### Plot ConnectionFraction
    if (data.__contains__('ConnectionFraction') and
       data.c.stats.__contains__('only_last')):   
        connections = np.zeros(N)
        lookback = 3000
        for i in range(N):
            for j in range(np.shape(data.ConnectionFraction)[0]):
                connections[i] += sum(data.ConnectionFraction[j][i]
                                           [-lookback:])/(lookback*1.0)
            connections[i] /= 1.0*np.shape(data.activity)[0]
        figure()
        plot(params,connections,'o')
        title('Fraction of ex-ex connections for last 3000 steps (%s)'
              %(data.c.stats.file_suffix[0]))
        xlabel('%s'%data.c.cluster.vary_param[0])
        ylabel('Connection fraction')
        utils.saveplot('Connections_%s.%s'
                        %(data.c.stats.file_suffix[0],ftype))
        
        figure()
        for i in range(N):
            #TODO average over all
            plot(data.ConnectionFraction[0][i])
        legend(data.paramtracker[0])
        xlabel('Steps')
        ylabel('Connection fraction')
        only_last = data.c.stats.only_last[0]
        N_steps = data.c.N_steps[0]
        stepsize = only_last//2
        xticks([0,stepsize,2*stepsize,3*stepsize,4*stepsize],
               [0,N_steps//2,'<--timelapse | last---------->',
               N_steps-only_last//2,N_steps])
        title('Connection fraction for %s = %.3f'
              %(data.c.cluster.vary_param[0],params[i]))
        utils.saveplot('Connections2_%s.%s'
                       %(data.c.stats.file_suffix[0],ftype))
                       
    if (data.__contains__('ConnectionFraction') and not
       data.c.stats.__contains__('only_last')): 
        figure()
        N_points = 1000
        spacing = data.c.steps_plastic[0]//N_points
        x = linspace(0,data.c.steps_plastic[0],N_points)
        for p in range(N_params):
            fractions = data.ConnectionFraction[:,p,
                                    :data.c.steps_plastic[0]:spacing]
            errorspan(x,mean(fractions,0),yerr=std(fractions,0),
                      label=params[p][0])
        xlim([x[0]-0.05*x[-1],x[-1]])
        legend(loc='upper right',title=param_name_plot)
        xlabel('Step')
        ylabel('Fraction of E-E connections')
        tight_layout()
        utils.saveplot('Connections_%s.%s'%\
                                    (data.c.stats.file_suffix[0],ftype))
    
    ### Plot effect of double_synapses
    if (data.__contains__('W_ee_history') and
            data.__contains__('W_ee_2_history')):
        W_ee_hs = data.W_ee_history
        W_ee2_hs = data.W_ee_2_history
        
        from plot_single import parallel_stats
        
        diffs = np.zeros((N_iterations,N_params,shape(W_ee_hs)[2]))
        cvs = np.zeros((N_iterations,N_params,shape(W_ee_hs)[2]))
        for (i) in range(N_params):
            for j in range(N_iterations):
                (diffs[j,i,:],cvs[j,i,:],_,_) = parallel_stats(
                                             W_ee_hs[j,i],W_ee2_hs[j,i])
                
        figure()
        x = linspace(0,data.c.N_steps[0],shape(W_ee_hs)[2])
        for (i,p) in enumerate(params):
            errorspan(x,mean(cvs[:,i],0),std(cvs[:,i],0),
                                      label=param_name_plot+" = %.2f"%p)
        plot([x[0],x[-1]],[0.083,0.083],'--k',
                                        label='CV from [Bartol et al.]')
        ylabel('Median CV between weight pairs')
        xlabel('Step')
        xlim([x[0]-0.05*x[-1],x[-1]])
        legend(loc='best')
        tight_layout()
        utils.saveplot('DoubleSynapses_CV_%s.%s'\
                        %(data.c.stats.file_suffix[0],ftype))
                        
    if data.__contains__('weefail'):
        weefail = data.weefail
        N_steps = data.c.N_steps[0]
        x = arange(N_steps)
        N_points = 1000
        spacing = data.c.steps_plastic[0]//N_points
        figure()
        for (i,p) in enumerate(params):
            errorspan(x[::spacing],mean(weefail[:,i,::spacing],0),
                      std(weefail[:,i,::spacing],0)/N_iterations,
                      label=param_name_plot+" = %.2f"%p)
        xlabel('Step')
        ylabel('Synaptic failure fraction')
        xlim([x[0]-0.05*x[-1],x[-1]])
        legend(loc='best')
        tight_layout()
        utils.saveplot('weefail_%s.%s'
                            %(data.c.stats.file_suffix[0],ftype))
                            
    ### Plot WeightLifetime
    if False and data.__contains__('WeightLifetime') and \
       any(data.WeightLifetime[0][:] > 0):
        figure()
        logbins = logspace(2,4,20)
        (y,_) = histogram(data.WeightLifetime[0]
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
        legend(('data','powerlaw-fit (a=%.3f k=%.3f)'
                %(popt[0],popt[1])))
        utils.saveplot('WeightLifetime_%s.%s'
                        %(data.c.stats.file_suffix[0],ftype))

    ### Plot WeightChangeStat
    if False and data.__contains__('WeightChange'):
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
        plt.suptitle('Change of Weights over %d Steps (%s)'
                     %(3000,data.c.stats.file_suffix[0]))
        utils.saveplot('WeightChange_%s.%s'
                       %(data.c.stats.file_suffix[0],ftype))
        
    ### Plot InputWeightStat
    if data.__contains__('InputWeight'):
        figure()
        N_samples = shape(data.InputWeight)[4]
        
        ## Different colors
        NUM_COLORS = N_params
        cm = plt.get_cmap('gist_rainbow')
        cNorm  = colors.Normalize(vmin=0, vmax=NUM_COLORS-1)
        scalarMap = mplcm.ScalarMappable(norm=cNorm, cmap=cm)
        plt.gca().set_color_cycle([scalarMap.to_rgba(i) for i in 
                                   range(NUM_COLORS)])

        sums_weights = np.zeros((N_params,N_iterations,N_samples))
        for param in range(N_params):
            for iteration in range(N_iterations):
                sums_weights[param,iteration,:] = \
                        data.InputWeight[iteration,param].sum(0).sum(0)
                sums_weights[param,iteration,:] /= \
                        sums_weights[param,iteration,0]
                sums_weights[param,iteration,
                             sums_weights[param,iteration]==0] = 1
        #average over iterations
        plot((sums_weights.sum(1)/(1.0*N_iterations)).T)
        xlabel('Step')
        only_last = data.c.stats.only_last[0]
        N_steps = data.c.N_steps[0]
        stepsize = only_last//2
        xticks([0,stepsize,2*stepsize,3*stepsize,4*stepsize],
               [0,N_steps//2,'<--timelapse | last---------->',
               N_steps-only_last//2,N_steps])
        ylabel('Normalized sum of all input weights')
        legend(data.paramtracker[0,:])
        title('Input weight influence with param %s'
               %data.c.cluster.vary_param[0])
        utils.saveplot('InputWeight_%s.%s'
                       %(data.c.stats.file_suffix[0],ftype))
        
    ### Plot SpontPatterns
    if data.__contains__('SpontPattern'):
        # gather or gatherv?
        if shape(shape(data.SpontPattern[:]))[0] == 2:
            gatherv = True
            index_old = 0
        else:
            N_indices = shape(data.SpontPattern)[3]
            gatherv = False
            indexfreqs_mean_cumm = np.zeros((N_params,N_indices))
            indexfreqs_std_cumm = np.zeros((N_params,N_indices))
            patternfreqs_mean_cumm = np.zeros((N_params,N_indices))
            patternfreqs_std_cumm = np.zeros((N_params,N_indices))
        
        for param in range(N_params):
            filename = os.path.join(pickle_dir,
                                    "source_plastic_%s_%.3f.pickle"
                                    %(param_name,params[param]))
            source_plastic = pickle.load(gzip.open(filename,"r"))
            if isinstance(source_plastic,TrialSource):
                source_plastic = source_plastic.source
            words = source_plastic.words
            word_string = ''.join(words)
            if gatherv:
                index_new = \
                       where(data.SpontPattern[:]==-1)[1][2*(param+1)-1]
                freqs = data.SpontPattern[:,index_old:index_new]
                index_old = index_new
                freqs = freqs[freqs>=0]
                freqs = reshape(freqs,(N_iterations,2,-1))
                N_indices = shape(freqs)[2]
            else:
                freqs = data.SpontPattern[:,param,:,:]
                
            # Normalize to relative frequencies
            freqs /= (1.*data.NormLast[0,param,0])
            
            # First index frequencies
            indexfreqs_mean = mean(freqs[:,0,:],0)
            indexfreqs_std= std(freqs[:,0,:],0)/sqrt(N_iterations)
            
            figure()
            x = arange(N_indices)
            bar(x,indexfreqs_mean,\
                yerr=indexfreqs_std,\
                align='center',label='Spontaneous freq.')
                #,color=repeat(['b','r'],[4,4]))
            #~ title('Spontaneous activity for %s=%.2f'
                  #~ %(param_name,params[param]))
            # this assumes transition probabilities independent of 
            # the precessor
            word_probs = source_plastic.probs[0]
            word_length = [len(x) for x in words]
            norm_probs = word_probs/sum(map(lambda x,y:x*y,
                                            word_probs,word_length))
            lstart = 0
            for (i,l) in enumerate(word_length):
                p = norm_probs[i]
                # default bar width is 0.8
                plot([lstart-0.4,lstart+l-0.6],[p,p],'r--')
                lstart += l
            plot([],[],'r--',label='Presentation freq.')
            xlim([-2,len(indexfreqs_mean)+1])
            ax = gca()
            ax.set_xticks(arange(len(word_string)))
            ax.set_xticklabels(array([x for x in word_string]))
            ylabel('Relative frequency')
            xlabel('Letter')
            tight_layout()
            legend(loc='best')
            utils.saveplot('SpontAct_%s_%s_%.2f.%s'\
                        %(data.c.stats.file_suffix[0],param_name_u,
                          params[param],ftype))
            
            # Then pattern frequencies
            # Normalize to relative occurances
            for i in range(N_iterations):
                freqs[i,1,:] /= sum(freqs[i,1,:])
            
            patternfreqs_mean = mean(freqs[:,1,:],0)
            patternfreqs_std = std(freqs[:,1,:],0)/sqrt(N_iterations)
            figure()
            N_patterns = len(words)*2
            bar(arange(N_patterns),patternfreqs_mean[:N_patterns],\
                yerr=patternfreqs_std[:N_patterns],align='center')
            #~ title('Spontaneous patterns for %s=%.2f'
                  #~ %(param_name,params[param]))
            xlim([-2,N_patterns+1])
            ylim([0,1])
            ax = gca()
            ax.set_xticks(arange(N_patterns))
            ax.set_xticklabels(words + [x[::-1] for x in words],
                               rotation=30,ha='right')
            ylabel('Relative frequency')
            xlabel('Pattern')
            tight_layout()
            utils.saveplot('SpontPat_%s_%s_%.2f.%s'
                           %(data.c.stats.file_suffix[0],param_name_u,
                             params[param],ftype))
                    
            if not gatherv:
                indexfreqs_mean_cumm[param,:] = indexfreqs_mean
                indexfreqs_std_cumm[param,:] = indexfreqs_std
                patternfreqs_mean_cumm[param,:] = patternfreqs_mean
                patternfreqs_std_cumm[param,:] = patternfreqs_std
        if not gatherv:
            figure()
            for index in range(N_indices):
                errorbar(params,#+random(shape(params))*0.1*std(params),
                         indexfreqs_mean_cumm[:,index],
                         yerr=indexfreqs_std_cumm[:,index],
                         label=word_string[index])
                hold('on')
            legend(loc='center right')
            #~ title('Letter frequencies')
            xlabel(param_name_plot)
            minmax = [min(params).copy(),max(params).copy()]
            delta = (minmax[1]-minmax[0])*0.1
            minmax[0] -= delta
            minmax[1] += delta
            xlim(minmax)
            ylabel('Relative frequency')
            tight_layout()
            utils.saveplot('change_freqs_%s.%s'%(param_name_u,ftype))
            
            figure()
            allwords = words + [x[::-1] for x in words]
            for index in range(N_patterns):
                errorbar(params,#+random(shape(params))*0.1*std(params),
                         patternfreqs_mean_cumm[:,index],
                         yerr=patternfreqs_std_cumm[:,index],
                         label=allwords[index])
                hold('on')
            legend(loc='center right')
            #~ title('Pattern frequencies')
            xlabel(param_name_plot)
            xlim(minmax)
            ylabel('Relative frequency')
            tight_layout()
            utils.saveplot('change_patterns_%s.%s'%(param_name_u,ftype))
    
    if data.__contains__('EvokedPred'):
        # Reps x Params x Words x Step x pinv/base
        max_step = shape(data.EvokedPred)[-2]#15 #-word_length
        pred_spont = data.EvokedPred[:,:,:,:,0]
        pred_base  = data.EvokedPred[:,:,:,:,1]
        for p in range(N_params):
            inputi = data.InputIndex[0,p]
            filename = os.path.join(pickle_dir,
                                    "source_%s_%s_%.3f.pickle"
                                    %(data.c.stats.quenching[0],
                                      param_name,params[p]))
            source = pickle.load(gzip.open(filename,"r"))
            if isinstance(source,TrialSource):
                source = source.source
            word_lengths = [len(x) for x in source.words]
            word_length = max(word_lengths)
                    
            figure()
            axvspan(0,word_length-1,color='#E6E6E6')
            secondstim_start = word_length # length of first word
            secondstim_stop = word_length # length of second word
            if data.c.stats.quenching[0] == 'test':
                secondstim_start += data.c.wait_min_test[0]
                secondstim_stop += data.c.wait_var_test[0]
            elif data.c.stats.quenching[0] == 'train':
                secondstim_start += data.c.wait_min_train[0]
                secondstim_stop += data.c.wait_var_train[0]
            else:
                secondstim_start = x.max()  # ugly and I know it
                secondstim_stop = x.max()+secondstim_start
            secondstim_stop += secondstim_start            
            axvspan(secondstim_start,secondstim_stop,facecolor='w',
                    edgecolor='#E6E6E6',
                    linewidth=0,hatch="x")
            from scipy.stats import nanmean
            pred_spont_p = nanmean(pred_spont[:,p,:,:max_step],1)
            pred_base_p = nanmean(pred_base[:,p,:,:max_step],1)
            x = arange(shape(pred_spont_p)[1])
            errorbar(x,mean(pred_spont_p,0),
                     std(pred_spont_p,0)/sqrt(N_iterations),color='b',
                     label='Spont. pred.')
            hold('on')
            errorbar(x,mean(pred_base_p,0),
                     std(pred_base_p,0)/sqrt(N_iterations),
                     color='#808080',label='Baseline')
            y_lim = ylim()
            ylim(y_lim)
            xlim([x.min(),x.max()])
            legend(loc='best')
            xlabel('Step after stimulus onset')
            ylabel('Pearson correlation to evoked response')
            #~ suptitle('%s = %.2f'%(param_name,params[p]))
            tight_layout()
            utils.saveplot('evokedpred_%s_%s_%.2f.%s'
                           %(data.c.stats.file_suffix[0],param_name_u,
                             params[p],ftype))
        # Assuming identical word length for shaded areas
        figure()
        axvspan(0,word_length-1,color='#E6E6E6')
        axvspan(secondstim_start,secondstim_stop,facecolor='w',
                    edgecolor='#E6E6E6',
                    linewidth=0,hatch="x")
        # Roll to get frac_A=1 to front (A first letter in alphabet and
        # evokedpred sorted by letters)
        frac_A = roll(data.c.frac_A[0],1)
        for (i,frac) in enumerate(frac_A):
            errorbar(x,mean(pred_spont[:,:,i],1).mean(0),
                    mean(pred_spont[:,:,i],1).std(0)/sqrt(N_iterations),
                    label='%.2fA'%frac)
        ylabel('Pearson correlation to evoked response')
        xlabel('Step after stimulus onset')
        legend(loc='best')
        tight_layout()
        utils.saveplot('evokedpred_byword_%s.%s'
                           %(data.c.stats.file_suffix[0],ftype))
                             
    if data.__contains__('Bayes'):
        # Remove all-zero returned matrices and matrices with 
        # values >+-10 from failed SVD and values==0 from failed SVD
        from scipy.interpolate import interp1d
        bayes = np.zeros(shape(data.Bayes)[1:])
        bayes_std = np.zeros(shape(data.Bayes)[1:])
        for p in range(N_params):
            tmp   = []
            for i in range(N_iterations):
                if not (any(data.Bayes[i,p]>10) or 
                         any(data.Bayes[i,p]<-10) or 
                         all(data.Bayes[i,p] == 0)):
                    tmp.append(data.Bayes[i,p])
            assert(not tmp == [])
            bayes[p] = mean(array(tmp),0)
            bayes_std[p] = std(array(tmp),0)/sqrt(N_iterations)
                    
        frac_A = data.c.frac_A[0]
        
        '''
        Linearly interpolate the crossing point between curve Y1 and Y2
        This assumes that Y1 starts of smaller than Y2
        It will return the first intersection point
        If there are no intersections, return the x-value at the end
        of the interval, where Y1 and Y2 are most similar
        '''
        def get_crossing(x,Y1,Y2,N_points=1000):
            precise_x = np.linspace(x.min(),x.max(),N_points)
            f_y1 = interp1d(x,Y1)
            f_y2 = interp1d(x,Y2)
            y_y1 = f_y1(precise_x)
            y_y2 = f_y2(precise_x)
            crossing = where(y_y1>y_y2)
            if shape(crossing)[1]>0:
                crosspoint = crossing[0][0]
            else:
                if abs((Y1[-1]-Y2[-1])) < abs((Y1[0]-Y2[0])):
                    crosspoint = N_points-1
                else:
                    crosspoint = 0
            return precise_x[crosspoint]
            
        raw_crossings = zeros((N_params,N_iterations))
        for i in range(N_params):
            for j in range(N_iterations):
                raw_crossings[i,j] = get_crossing(frac_A,
                            data.Bayes[j,i,:,4],data.Bayes[j,i,:,5])
        crossings = mean(raw_crossings,1)
        crossings_std = std(raw_crossings,1)
        
        # Fit optimal model    
        from chartmann.spont.optimal_channels import OptimalChannels
        channels = OptimalChannels(N_u=data.c.N_u_e[0])
        N_As = (frac_A*data.c.N_u_e[0]).round().astype(int)
        
        def opt_wrapper(x,p_uA_given_A,p_uA_given_B,p_A):
                channels.p_uA_given_A = p_uA_given_A
                channels.p_uB_given_B = p_uA_given_A
                channels.p_uA_given_B = p_uA_given_B
                channels.p_uB_given_A = p_uA_given_B
                return channels.optimal_inference(p_A=p_A,N_As=x,
                                                  N_samples=10000)
        
        N_optprobs = int(round(0.9/0.05))+1
        ps_uA_given_A = linspace(0.05,0.95,N_optprobs)
        ps_uA_given_B = linspace(0.05,0.95,N_optprobs)
        best_so_far = inf
        '''
        Parameter symmetries:
        if:
          p_uA_given_A = a
          p_uA_given_B = b
        then the following combinations give the same result:
          p_uA_given_A = b
          p_uA_given_B = a
        and
          p_uA_given_A = 1-a
          p_uA_given_B = 1-b
        Intuitions for conservation of information:
          1-  -> just interpret the transmission as success when failed
          b=a -> just renaming of variables
        '''
        for pAA in ps_uA_given_A:
            for pAB in ps_uA_given_B[ps_uA_given_B<=pAA]:
                dists = zeros((N_params,N_iterations))
                for i in range(N_params):
                    y_est = opt_wrapper(N_As,pAA,pAB,params[i])
                    for j in range(N_iterations):
                        # least squares
                        dists[i,j] = np.linalg.norm(data.Bayes[j,i,:,4]-y_est)**2 
                dist = mean(dists)
                if dist<best_so_far:
                    p_uA_given_A = pAA
                    p_uA_given_B = pAB
                    best_so_far = dist
                    
        #~ p_uA_given_A = 0.3
        #~ p_uA_given_B = 0.05

        fitted_channels = OptimalChannels(p_uA_given_A=p_uA_given_A,
                                          p_uA_given_B=p_uA_given_B,
                                          N_u=data.c.N_u_e[0])
        
        opt_posteriors = zeros((N_params,len(frac_A)))
        opt_crossings = zeros(N_params)
        for i in range(N_params):
            # Many samples for pretty plots
            opt_posteriors[i,:] = fitted_channels.optimal_inference(
                                 p_A=params[i],N_As=N_As,N_samples=10000)
            opt_crossings[i] = get_crossing(frac_A,opt_posteriors[i],
                                            1-opt_posteriors[i])
            
            
        for i in range(N_params):
            fig, ax = plt.subplots()
            errorbar(frac_A,bayes[i,:,0],bayes_std[i,:,0],fmt='-b',
                     label='Decision A')
            hold('on')
            errorbar(frac_A,bayes[i,:,1],bayes_std[i,:,1],fmt='-g',
                     label='Decision B')
            ylim([0,1])
            xlim([0,1])
            #~ title('%s = %.2f'%(param_name,params[i]))
            tmp = 1-params[i]
            plot([tmp,tmp],[0,1],'--k',label='1-prior(A)')
            legend(loc='upper center')
            xlabel('Fraction of cue A in ambiguous cue')
            ylabel('Output gain +- stderr')
            
            
            utils.saveplot('bayes_drive_%s_%f.%s'%(param_name_u,
                           params[i],ftype))
            figure()
            # Lines for optimality explanation before data for overlap
            # Old/wrong optimal lines
            #~ tmp = 1-params[i]
            #~ plot([tmp,tmp],[0,1],'--k',label='1-prior(A)')
            #~ hold('on')
            #~ denom = frac_A*params[i]+frac_A[::-1]*(1-params[i])
            #~ plot(frac_A,frac_A*params[i]/denom,'-', color='#808080', \
                                                        #~ label='Optimal')
            #~ plot(frac_A,frac_A[::-1]*(1-params[i])/denom,'-',\
                                                        #~ color='#808080')

            plot(frac_A,opt_posteriors[i],'--', color='#808080',
                                                        label='Prob. model')
            plot(frac_A,1-opt_posteriors[i],'--', color='#808080')
            # Actual data here
            errorbar(frac_A,bayes[i,:,4],bayes_std[i,:,4],fmt='-b',\
                                                     label='Decision A')
            hold('on')
            errorbar(frac_A,bayes[i,:,5],bayes_std[i,:,5],fmt='-g',\
                                                     label='Decision B')
            ylim([0,1])
            xlim([0,1])
            #~ title('%s = %.2f'%(param_name,params[i]))
            # Reorder labels b/c ugly
            ax = gca()
            handles, labels = ax.get_legend_handles_labels()
            labels = [z for z in array(labels)[[1,2,0]]]
            handles = [z for z in array(handles)[[1,2,0]]]
            
            leg = ax.legend(handles, labels, loc='best')
            leg.get_frame().set_alpha(0.5)
            
            #~ legend(loc='best')
            xlabel('Fraction of A in ambiguous stimulus')
            ylabel('Fraction of decisions')
            #~ if i < (N_params-1):
                #~ utils.saveplot('bayes_dec_%s_%f.%s'
                               #~ %(param_name_u,params[i],ftype))
            utils.saveplot('bayes_dec_frac_%s_%f.%s'
                           %(param_name_u,params[i],ftype))

        figure()
        plot(1-params[:,0],opt_crossings,'--', color='#808080',
             label='Prob. model')
        errorbar(1-params[:,0],crossings,
                 crossings_std/sqrt(N_iterations),fmt='o-',
                 label='Intersection')
        #~ plot([tmp,tmp],[0,1],'--k')
        # Reorder labels b/c ugly
        ax = gca()
        handles, labels = ax.get_legend_handles_labels()
        labels = [x for x in array(labels)[[1,0]]]
        handles = [x for x in array(handles)[[1,0]]]
        leg = ax.legend(handles, labels, loc='best')
        leg.get_frame().set_alpha(0.5)
        ylim([0,1])
        xlim([0,1])
        xlabel('1 - ('+param_name_plot+')')
        ylabel('Intersection of decisions')
        tight_layout() # for suplot spacing
        utils.saveplot('bayes_dec_intersect_%s.%s'
                       %(param_name_u,ftype))
                       
        figure()
        title('Fitted Parameters')
        text(2,7,'p(transmission|input) = %.2f'%p_uA_given_A,
             fontsize=20)
        text(2,3,'p(transmission|noinput) = %.2f'%p_uA_given_B,
             fontsize=20)
        ylim([0,10])
        xlim([0,10])
        ax = gca()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        utils.saveplot('parameters_channelmodel_%s.%s'%(param_name_u,
                                                        ftype))                                                        
        #~ import ipdb; ipdb.set_trace()
    if data.__contains__('SpontBayes'):
        sb = data.SpontBayes
        # over all conditions: check if higher-than-mean readout
        # corresponds to higher-than-mean activation of input units
        mean_readout = mean(mean(sb,0),2)[:,:,2:]
        mean_act = mean(mean(sb,0),2)[:,:,:2]
        n_conditions = shape(sb)[2]
        relative_effect = np.zeros((N_params,n_conditions,2))
        excess = np.zeros((N_params,2))
        for param in range(N_params):
            
            for i in range(n_conditions):
                indices_0 = where(sb[:,param,i,:,2]
                                  >mean_readout[param,i,0])
                indices_1 = where(sb[:,param,i,:,3]
                                  >mean_readout[param,i,1])
                # ugly mean computation
                vals_0 = []
                vals_1 = []
                for j in range(shape(indices_0)[1]):
                    vals_0.append(sb[indices_0[0][j],param,i,
                                  indices_0[1][j],0])
                for j in range(shape(indices_1)[1]):
                    vals_1.append(sb[indices_1[0][j],param,i,
                                  indices_1[1][j],1])
                relative_effect[param,i,0] = mean(array(vals_0))\
                                              /mean_act[param,i,0]
                relative_effect[param,i,1] = mean(array(vals_1))\
                                              /mean_act[param,i,1]
                                                          
            excess[param,0] = mean((mean_act[param,:,0]-
                                    frac_A*data.c.N_u_e[0]))
            excess[param,1] = mean((mean_act[param,:,1]-
                                    frac_A[::-1]*data.c.N_u_e[0]))
        figure()
        boxplot(relative_effect.flatten()*100-100)
        hold('on')
        plot([0.75,1.25],[0,0],'--k')
        title('Effect of above-average readout on input activity')
        ylabel('Increased input activity  [%]')
        xlabel('Collapsed over all values of the %s'%param_name_plot)
        xticks([])
        xlim([0.75,1.25])
        utils.saveplot('spontbayes_box_%s_%f.%s'
                       %(param_name_u,params[param],ftype))
        figure()
        plot(params,excess[:,0],'-b',label='A units')
        hold('on')
        plot(params,excess[:,1],'-g',label='B units')
        xlim([0,1])
        legend(loc = 'upper center')
        xlabel(param_name_plot)
        ylabel('Mean excess activity over all stimuli')
        utils.saveplot('spontbayes_excess_%s_%f.%s'
                       %(param_name_u,params[param],ftype))
                                            
    if data.__contains__('TrialBayes'):
        filename = os.path.join(pickle_dir,"source_%s_%s_%.3f.pickle"\
                                        %('test',param_name,params[0]))
        source = pickle.load(gzip.open(filename,"r"))
        if isinstance(source,TrialSource):
            source = source.source
        word_lengths = [len(x) for x in source.words]
        word_length = max(word_lengths)
        agreements = np.zeros((N_iterations*N_params,2,\
                               shape(data.TrialBayes)[3]))
        count = 0
        ig = 30 # only use ig step
        # for the time being:
        forward_pred = data.c.stats.forward_pred[0]
        x = arange(-ig+forward_pred,forward_pred)
           
        for i in range(N_params):
            tmp = data.TrialBayes[:,i,0,:]*100
            nonzero_set = set(where(tmp!=-100)[0])
            nonzero_list = [n for n in nonzero_set]
            trials = len(nonzero_list)
            tmp = tmp[nonzero_list]
            agreements[count:count+trials,0,:] = tmp
            tmp = tmp[:,-ig:] 
            agreement = mean(tmp,0)
            agreement_sem = std(tmp,0)/sqrt(trials)
            tmp_lstsq = data.TrialBayes[:,i,1,:]*100
            tmp_lstsq = tmp_lstsq[nonzero_list]
            agreements[count:count+trials,1,:] = tmp_lstsq
            tmp_lstsq = tmp_lstsq[:,-ig:] # ignore the last stim
            agreement_lstsq = mean(tmp_lstsq,0)
            agreement_lstsq_sem = std(tmp_lstsq,0)/sqrt(trials)
            count += len(nonzero_list)
            figure()
            errorbar(x,agreement,agreement_sem,color='#808080',
                     label='Baseline')
            hold('on')
            errorbar(x,agreement_lstsq,agreement_lstsq_sem,color='b',
                     label='Spont. prediction')
            y_lim = ylim()
            x_lim = xlim()
            axvspan(0,word_length-1,color='#E6E6E6')
            plot([word_length-1,word_length-1],y_lim,'--g',
                 label='Pred. position')
            ylim(y_lim)
            xlim(x_lim)
            xlabel('Step relative to stimulus onset')
            title('%s = %.2f'%(param_name_plot,params[i]))
            ylabel('Correct predictions [%]')        
            utils.saveplot('trialbayes_%.2f_%s.%s'\
                        %(params[i],data.c.stats.file_suffix[0],ftype))
        
        agreements = agreements[:count,:,-ig:]
        figure()
        errorbar(x,mean(agreements[:,0,:],0),
                 std(agreements[:,0,:],0)/sqrt(count),color='#808080',
                 label='Baseline')
        errorbar(x,mean(agreements[:,1,:],0),
                 std(agreements[:,1,:],0)/sqrt(count),color='b',
                 label='Spont. prediction')  
        y_lim = ylim()          
        axvspan(0,word_length-1,color='#E6E6E6')
        plot([word_length-1,word_length-1],y_lim,'--g',
                 label='Pred. position')
        legend(loc='upper left')
        xlabel('Step relative to stimulus onset')
        ylabel('Correct predictions [%]')
        utils.saveplot('trialbayes_average_%s.%s'\
                        %(data.c.stats.file_suffix[0],ftype))

                        
    # quenching variability
    if data.__contains__('Spikes') and \
        data.c.stats.quenching is not None:
        spikes_before = 10
        spikes_after = 10
        # number of bins left and right of t (2 -> boxwidth=5)
        window_width = data.c.stats.quenching_window[0] 
        weighted_regression = True
        mode = data.c.stats.quenching[0]
        assert(mode == 'train' or mode == 'test')
        
        # Get N_words for array
        filename = os.path.join(pickle_dir,"source_%s_%s_%.3f.pickle"\
                                        %(mode,param_name,params[0]))
        source = pickle.load(gzip.open(filename,"r"))
        if isinstance(source,TrialSource):
            source = source.source
        word_lengths = [len(x) for x in source.words]
        max_word_length = max(word_lengths)
        N_words = len(source.words)
        total_length = max_word_length + spikes_before + spikes_after
        # Look at last half of training set
        steps_plastic = data.c.steps_plastic[0]
        steps_noplastic_train = data.c.steps_noplastic_train[0]
        steps_noplastic_test = data.c.steps_noplastic_test[0]
        if mode == 'train':
            interval = [-steps_noplastic_train-steps_noplastic_test,
                    -steps_noplastic_test]
        else: # test because of assert
            interval = [steps_plastic+steps_noplastic_train,-1]
        
        # same order as all: first it, then params
        FF = np.zeros((N_iterations,N_params,N_words,
                       total_length-2*window_width))
        means = np.zeros((N_iterations,N_params,N_words,
                          total_length-2*window_width))
        allvars = np.zeros((N_iterations,N_params,N_words,
                            total_length-2*window_width))
        if matlab_comparison:
            try:
                from mlabwrap import mlab
            except ImportError:
                matlab_comparison = False
        if matlab_comparison:
            mlab.addpath(
                    '/home/chartmann/Desktop/sorn/py/chartmann/spont/')
            FFs_mlab = np.zeros((N_iterations,N_params,total_length-7))
            means_mlab = np.zeros((N_iterations,N_params,
                                   total_length-7))
            meansAll_mlab = np.zeros((N_iterations,N_params,
                                      total_length-7))

        for p in range(N_params):
            for i in range(N_iterations):                
                input_spikes = data.Spikes[i,p][:,
                                                interval[0]:interval[1]]
                input_index = data.InputIndex[i,p][
                                                interval[0]:interval[1]]
                
                # Determine minimum number of trials
                min_trials = inf
                word_start = 0
                for j in range(N_words):
                    indices = find(input_index==word_start)
                    tmp_trials = sum((indices >= spikes_before)*\
                                 (indices <= shape(input_index)[0]
                                             -spikes_after))
                    if tmp_trials < min_trials:
                        min_trials = tmp_trials
                    word_start += word_lengths[j]
                
                # build trial matrix (condition x trial x t x spikes)
                N_e = shape(input_spikes)[0]
                trials = np.zeros((N_words,min_trials,total_length,N_e))
                word_start = 0
                for word in range(N_words):
                    indices = find(input_index==word_start)
                    indices = indices[((indices >= spikes_before) * 
                                       (indices <= shape(input_index)[0]
                                                   -(spikes_after
                                                     +max_word_length))
                                      )]
                    indices = indices[-min_trials:] # take from end
                    for (trial,j) in enumerate(indices):
                        trials[word,trial,:,:] = input_spikes[:,
                                                  j-spikes_before:j
                                                  +max_word_length
                                                  +spikes_after].T
                    word_start += word_lengths[word]
                
                # Determine units that do not receive input
                noinput_units = arange(N_e)[data.InputUnits[i,p]==0]
                
                if matlab_comparison:
                    result = mlab.VarVsMean_pythontomat_bulk(trials[:,:,
                                                       :,noinput_units])
                    N = 1
                    FFs_mlab[i,p] = result[:,:N].T
                    means_mlab[i,p] = result[:,3*N:4*N].T/1000.
                    meansAll_mlab[i,p] = result[:,7*N:8*N].T/1000.
                
                for word in range(N_words):
                    for (t_i,t) in enumerate(arange(0,
                                            total_length-2*window_width)):
                        # Take this procedure from quenching variability
                        # paper figure 4:
                        # Regress between means and variances for all 
                        # neurons in small interval (in our case in 
                        # single step) over trials
                        
                        # This is summing over the window
                        # This indexing reshapes to
                        # (neurons x trials x window)
                        count = sum(trials[word,:,t:
                                      t+2*window_width+1,noinput_units],2)
                        meanss = mean(count,1)              
                        means[i,p,word,t_i] = mean(meanss)
                        varss = std(count,1)**2 
                        allvars[i,p,word,t_i] = mean(varss)
                        weighting = eye(shape(meanss)[0])
                        if weighted_regression:
                            for j in range(shape(meanss)[0]):
                                weighting[j,j] = min_trials/\
                                                  ((meanss[j]+0.001)**2)
                        slope = np.dot(np.dot(meanss.T,weighting),\
                                       varss)/np.dot(meanss.T,\
                                               np.dot(weighting,meanss))
                        FF[i,p,word,t_i] = slope
                                
            x = linspace(-spikes_before+2*window_width,
                      spikes_after+max_word_length-1,
                      total_length-(window_width*2))
            
            if matlab_comparison:
                x_mlab = x[:shape(FFs_mlab)[2]]
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
                mmeanall = mean(meansAll_mlab[:,p],0)
                smeanall = std(meansAll_mlab[:,p],0)/sqrt(N_iterations)
                mmean = mean(means_mlab[:,p],0)
                smean = std(means_mlab[:,p],0)/sqrt(N_iterations)
                mFF = mean(FFs_mlab[:,p],0)
                sFF = std(FFs_mlab[:,p])/sqrt(N_iterations)
                
                errorbar(x_mlab,mmeanall,yerr=smeanall,c='0.5',
                         label="Raw",lw=lw)
                errorbar(x_mlab,mmean,yerr=smean,fmt='k',
                         label="'Matched'",lw=lw)
                minmax = [min(hstack((mmeanall,mmean))), 
                          max(hstack((mmeanall,mmean)))]
                minmax[0] = round(minmax[0]-0.0049,3)
                minmax[1] = round(minmax[1]+0.0049,3)
                minmaxx = [x_mlab[0]-1,max(x_mlab)+0.2]
                ylabel('Spikes/step')
                gca().locator_params(axis='y',nbins=4) # 4 ticks/axis
                remove_axes(gca())
                legend(loc='best')
                ylim(minmax)
                xlim(minmaxx)
                tight_layout()
                
                subplot(gs[1])
                plot(x_mlab,mFF,'k',label='FF',lw=lw)
                plot(x_mlab,mFF-sFF,c='0.5',label='SEM',lw=lw)
                plot(x_mlab,mFF+sFF,c='0.5',lw=lw)
                quiver(-3,ylim()[0],0,0.1,scale=1,label='Stim on')
                gca().locator_params(axis='y',nbins=4) # 4 ticks/axis
                remove_axes(gca())
                legend(loc='best')
                ylabel('Fano Factor')
                ylim([min(mFF-sFF)-0.01,max(mFF+sFF)+0.01])
                xlim(minmaxx)
                tight_layout()
                utils.saveplot('quenching_word_%s_%s_%.2f.%s'
                               %(data.c.stats.file_suffix[0],
                                 param_name_u,params[p],ftype))
                

            FF[isnan(FF)] = 0 # nans usually are around small values

            
        # Rearrange to match frequency
        FFnew = roll(FF,-1,axis=2)
        for p in range(N_params):
            for word in range(N_words):
                # This is AX* and BX* (word starting with index 0 and 
                # index word_length because A,B first two letters in 
                # alphabet)
                # This is accounted for in the Bayes stat by resorting
                if word == 0 or word==1: 
                    fig,axes = subplots(2, sharex=True)
                    ax1 = axes[1]
                    ax1.errorbar(x,mean(allvars[:,p,word,:],0),
                                 std(allvars[:,p,word,:],0)\
                                 /sqrt(N_iterations),fmt='b')
                    ax1.hold('on')
                    ax1.set_xlabel('Step')
                    ax1.yaxis.label.set_color('b')
                    y_lim = [min(flatten(mean(allvars[:,:,word,:],0))),
                             max(flatten(mean(allvars[:,:,word,:],0)))]
                    ax1.set_ylim(y_lim)
                    locator_params(axis='y',nbins=4) # 4 ticks/axis
                    ax2 = ax1.twinx()
                    ax2.errorbar(x,mean(means[:,p,word,:],0),
                                 std(means[:,p,word,:],0)\
                                 /sqrt(N_iterations),fmt='r')
                    ax2.yaxis.label.set_color('r')
                    ax2.set_ylabel('Mean rate')
                    locator_params(axis='y',nbins=4) # 4 ticks/axis
                    y_lim = [min(flatten(mean(means[:,:,word,:],0))),
                             max(flatten(mean(means[:,:,word,:],0)))]
                    hold('on')
                    ax1.axvspan(0,word_lengths[word]-1,color='#E6E6E6')
                    ylim(y_lim)
                    xlim([x.min(),x.max()])
                    ax1.set_ylabel('Variance')
                    
                    ax = axes[0]
                    ax.errorbar(x,mean(FF[:,p,word,:],0),
                                std(FF[:,p,word,:],0)/sqrt(N_iterations)
                                ,fmt='k')
                    ax.set_ylabel('Fano factor')
                    ax.locator_params(axis='y',nbins=4) # 4 ticks/axis
                    # yaxis identical for all parameters for each word
                    y_lim = [min(flatten(mean(FF[:,:,word,:],0))),
                             max(flatten(mean(FF[:,:,word,:],0)))] 
                    hold('on')
                    ax.axvspan(0,word_lengths[word]-1,color='#E6E6E6')
                    ax.set_ylim(y_lim)
                    ax.legend(loc='lower left')
                                                       
                    tight_layout()
                    # because tight_layout doesn't recognize twinx
                    fig.subplots_adjust(right=0.9) 
                    utils.saveplot('quenching_word_%d_%s_%s_%.2f.%s'
                                   %(word,data.c.stats.file_suffix[0],
                                     param_name_u,params[p],ftype))
            # suptitle('%s = %.2f'%(param_name,params[p]))

            # Plot ambiguity vs. FF for each condition
            if False:
                minFFs = mean(FFnew.min(axis=3)[:,p],0)
                stdminFFs = std(FFnew.min(axis=3)[:,p],0)/sqrt(N_iterations)
                figure()
                errorbar(frac_A,minFFs,stdminFFs,label='Fano factor')
                y_lim = ylim()
                axvline(1-params[p],color='k',linestyle='dashed',
                        label='1-prior(A)')
                ylim(y_lim)
                gca().locator_params(axis='y',nbins=4) # 4 ticks/axis
                xlabel('Fraction of A in ambiguous stimulus')
                ylabel('Minimal Fano factor')
                legend(loc='best')
                tight_layout()
                xlim([-0.02,1.02])
                utils.saveplot('quenching_vs_amb_%s_%s_%.2f.%s'
                               %(data.c.stats.file_suffix[0],param_name_u,
                                 params[p],ftype))
        
        if False:
            # Plot prior vs. max. FF
            # For each stimulation condition:
            #   For each iteration, take the word 
            #       that maximizes the minimal FF
            #   Then average over these words
            frac_range = frac_A[-1]-frac_A[0]
            averagefrac = mean(argmax(FFnew.min(axis=3),2)
                               /((len(frac_A)-1)/frac_range),0)
            stdfrac = std(argmax(FFnew.min(axis=3),2)
                         /((len(frac_A)-1)/frac_range),0)/sqrt(N_iterations)
            # Assume even spacing of frac_A
            offset = frac_A[0]
            averagefrac += offset
            figure()
            plot([frac_A[0],frac_A[-1]],[frac_A[0],frac_A[-1]],
                 color='#808080',label='Identity')
            # Reverse to match 1-(prior(A))
            errorbar(params,averagefrac[::-1],stdfrac[::-1],fmt='o-',
                     label='Fraction')
            xlabel('1 - ('+param_name_plot+')')
            ylabel('Fraction of A with highest variability')
            legend(loc='best')
            tight_layout()
            xlim([0,1])
            ylim([0,1])
            utils.saveplot('queisi_snching_vs_prior_%s.%s'
                           %(data.c.stats.file_suffix[0],ftype))
                                                
    if data.__contains__('AttractorDynamics'):
        frac_A = data.c.frac_A[0]
        for p in range(N_params):
            output_dists = data.AttractorDynamics
            figure()
            # This is now frac x step (from cue to target)
            mean_od = mean(output_dists[:,p,:,:],0).T[:,::-1]
            std_od = std(output_dists[:,p,:,:],0).T[:,::-1]\
                     /sqrt(N_iterations)
            x = arange(-shape(mean_od)[1]+1,1)
            for (i,frac) in enumerate(frac_A):
                errorbar(x,mean_od[i,:],std_od[i,:],label="%.2f"%frac)
            ylabel('Distance between output gains')
            xlabel('Steps before target')
            legend()
            utils.saveplot('attractordynamics_%s_%s_%.2f.%s'
                           %(data.c.stats.file_suffix[0],param_name_u,
                             params[p],ftype))
                            
    if data.__contains__('OutputDist'):
        output_dist = data.OutputDist[:,:,0,:]
        output_std  = data.OutputDist[:,:,1,:]
        frac_A = data.c.frac_A[0]
        for i in range(N_params):
            figure()
            errorbar(frac_A, mean(output_dist[:,i,:],0), 
                                 std(output_dist[:,i,:],0), fmt='o-')
            ylim([0,1])
            x_lim = xlim()
            xlim([x_lim[0]-0.1,x_lim[1]+0.1])
            xlabel('Fraction of A in ambiguous stimulus')
            ylabel('Mean abs diff of normalized output gain +- std')
            title('%s = %.2f - mean(min) = %.2f'
                   %(param_name_plot,params[i],
                     # get min for each trial and av.
                     mean(output_dist[:,i,:].min(1))))
            utils.saveplot('outputdist_%s_%s_%.2f.%s'
                           %(data.c.stats.file_suffix[0],param_name_u,
                             params[i],ftype))
        figure()
        errorbar(params,mean(mean(output_dist,2),0),
                 std(mean(output_dist,2),0)/sqrt(N_iterations),fmt='o-')
        x_lim = xlim()
        xlim([x_lim[0]-0.1,x_lim[1]+0.1])
        xlabel(param_name_plot)
        ylabel('Attractor score')
        utils.saveplot('attractor_%s_%s_%.2f.%s'
                        %(data.c.stats.file_suffix[0],param_name_u,
                          params[i],ftype))
                          
    # Plot evoked pred vs. FF (high FF should yield better ep)
    # first normalize each iteration and param
    if data.__contains__('EvokedPred') and 'FF' in locals():
        diff = pred_spont[:,:,:,1] - pred_base[:,:,:,1]
        FFs = FF[:,:,:,11]
        for p in range(N_params):
            for i in range(N_iterations):
                diff[i,p] -= diff[i,p].min()
                diff[i,p] /= diff[i,p].max()
                FFs[i,p]  -= FFs[i,p].min()
                FFs[i,p]  /= FFs[i,p].max()
        FFs = FFs.flatten()
        diff = diff.flatten()
        figure()
        scatter(FFs,diff)
        (s,p) = stats.pearsonr(FFs,diff)
        xlabel('Normalized Fano factor after stimulus onset')
        ylabel('Normalized(spontpred - staticpred)')
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
        utils.saveplot('evokedpred_FF_%s.%s'\
                        %(data.c.stats.file_suffix[0],ftype))
        
        # Finally plot EP vs. condition at step 1 (indisting. at step0)
        # look at method for how predictions are sorted
        # --> almost sorted, but not averaged etc.

        if data.c.stats.quenching[0] == 'test' and False:
            frac_A = data.c.frac_A[0]
            for p in range(N_params):
                pred_p = pred_spont[:,p,:,1]
                # can mean here because order of means doesn't matter
                pred_p = mean(pred_p,0) # over iterations
                to_mean = pred_p[2:]
                meaned = [mean([x,y]) for (x,y) in zip(to_mean[::2],
                          to_mean[1::2])]
                # B, C, D, ..., A
                pred_p = hstack((pred_p[1],array(meaned),pred_p[0]))
                pred_s = pred_base[:,p,:,1]
                pred_s = mean(pred_s,0)
                to_mean = pred_s[2:]
                meaned = [mean([x,y]) for (x,y) in zip(to_mean[::2],
                          to_mean[1::2])]
                pred_s = hstack((pred_s[1],array(meaned),pred_s[0]))
                figure()
                plot(frac_A,pred_p,label='Pinv')
                hold('on')
                plot(frac_A,pred_s,label='STA')
                xlabel('Fraction of A in ambiguous stimulus')
                ylabel('Prediction')
                suptitle('%s = %.2f'%(param_name_plot,params[p]))
                legend()
                utils.saveplot('evokedpred_fracA_%s_%s_%.2f.%s'
                               %(data.c.stats.file_suffix[0],
                                 param_name_u,params[p],ftype))
    
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
            kl = np.sum(np.where(p != 0, p * np.log2(p / q), 0))
            return kl
        
        kl_evoked1_spont = zeros((N_params,N_iterations))
        kl_spont_evoked1 = zeros((N_params,N_iterations))
        kl_evoked_12 = zeros((N_params,N_iterations))
        kl_evoked_21 = zeros((N_params,N_iterations))
        kl_spont_12 = zeros((N_params,N_iterations))
        kl_spont_21 = zeros((N_params,N_iterations))
        
        kl_exp_spont = zeros((N_params,N_iterations))
        kl_con_spont = zeros((N_params,N_iterations))
        
        for p in range(N_params):
            for i in range(N_iterations):
                p_evoked_1 = data.patternprobability[i,p][0]
                p_evoked_2 = data.patternprobability[i,p][1]
                p_spont_1 = data.patternprobability[i,p][2]
                p_spont_2 = data.patternprobability[i,p][3]
                p_spont = (p_spont_1+p_spont_2)/2
                
                kl_evoked1_spont[p,i] = KL(p_evoked_1,p_spont)
                kl_spont_evoked1[p,i] = KL(p_spont,p_evoked_1)
                
                kl_evoked_12[p,i] = KL(p_evoked_1,p_evoked_2)
                kl_evoked_21[p,i] = KL(p_evoked_2,p_evoked_1)
                kl_spont_12[p,i] = KL(p_spont_1,p_spont_2)
                kl_spont_21[p,i] = KL(p_spont_2,p_spont_1)
                
                kl_exp_spont[p,i] = KL(p_evoked_1,p_spont)
                kl_con_spont[p,i] = KL(p_evoked_2,p_spont)
            
            figure()
            bar([1,2,3],[mean(kl_evoked1_spont[p]),mean(kl_evoked_12[p]),
                         mean(kl_spont_12[p])],yerr=[
                         std(kl_evoked1_spont[p]),std(kl_evoked_12[p]),
                         std(kl_spont_12[p])],align='center')
            xticks([1,2,3],['$D(e||s)$','$D(e||e)$','$D(s||s)$'])
            ylabel('KL-Divergence')
            title('%s = %s'%(param_name_u,params[p]))
            xlim([0.5,3.5])
            utils.saveplot('KLdiv_%s_%s_%.2f.%s'
                           %(data.c.stats.file_suffix[0],param_name_u,
                             params[p],ftype))
        figure()
        x = arange(len(params))
        bar(x,mean(kl_evoked1_spont,1),
            yerr=std(kl_evoked1_spont,1)/sqrt(N_iterations), 
            align='center')
        xticks(x,['%d'%p for p in params],rotation=30,ha='right')
        ylabel('KL-Divergence $D(e||s)$')
        xlabel(param_name_plot)
        utils.saveplot('KLdiv_%s.%s'\
                            %(data.c.stats.file_suffix[0],ftype))
        
        # Figure assuming first and second half of evoked are 
        # experiment and control, respectively
        figure()
        x = arange(len(params)*2)[::2]
        dx = 0.4
        bar(x-dx,mean(kl_exp_spont,1),
            yerr=std(kl_exp_spont,1)/sqrt(N_iterations), 
            align='center',color='r',linewidth=2,ecolor='k',
            label='Natural')
        bar(x+dx,mean(kl_con_spont,1),
            yerr=std(kl_con_spont,1)/sqrt(N_iterations), 
            align='center',color='g',linewidth=2,ecolor='k',
            label='Control')
        for p in range(N_params):
            label_diff(x[p]-dx,x[p]+dx,kl_exp_spont[p],
                       kl_con_spont[p],gca())
        xticks(x[::2],[' %d'%(p//1000) for p in params[::2]],
               ha='center')
        ylabel('KL-Divergence $D(e||s)$')
        legend(loc='best')
        if param_name == 'steps_plastic':
            param_name_plotting = 'Steps with plasticity [$*10^3$]'
        else:
            param_name_plotting = param_name
        xlabel(param_name_plotting)
        tight_layout()
        utils.saveplot('KLdiv_new_%s.%s'\
                            %(data.c.stats.file_suffix[0],ftype))
        
if __name__ == '__main__':
    plot_results(path, datafile)
    show()
