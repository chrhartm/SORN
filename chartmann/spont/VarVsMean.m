%
% usage: Result = VarVsMean(data, times, params)
%
% OVERVIEW
%   This function regresses the spike-count variance versus the spike-count mean 
%   and reports the slope (the Fano Factor).
%
%   This is done at multiple timepoints, which the user inputs.
%
%   Unless asked otherwise, the distribution of mean counts is matched (via dowselection),
%   so that all timepoints have the same distribution of mean counts.
%
% INPUTS
%   There are 2 required inputs: 'data', and 'times', and one optional input 'params'
%   
%   'DATA'
%           A structure, each element of which contains a field 'spikes' which is a matrix with 1 row 
%       per trial and 1 column per millisecond.  Each entry should be a '1' (there was a spike in 
%       that ms on that trial) or a '0' (there wasn't).  All data should be time-aligned to some event.
%           For example, suppose you collect 100 trials, each with 200 ms of pre-target
%       data and 800 ms of post-target data. Your 'spikes' matrix would then be 100 by 1000 
%       (with one of these for each element of 'data').
%       E.G.               
%           trial 1     0 0 0 1 0 0 0 1 0 0 1 ... 0 1 0 0 0 1
%           trial 2     0 0 1 0 0 1 0 0 0 1 0 ... 0 0 1 0 1 0 
%           trial 3     0 0 0 0 1 1 0 1 0 0 0 ... 0 1 0 0 0 0 
%             ...
%           trial n     1 0 1 1 0 0 0 1 0 0 0 ... 0 0 0 1 1 0 
%
%           Running this analysis only makes sense if you collected data for many neurons &/or many conditions.
%       Each element of 'data' should contain all the trials for one neuron/condition combination.  Thus, if you
%       collected data from 100 neurons using 10 conditions, 'data' would be length 1000.
%       **  The field containing the spike matrix MUST be named 'spikes'. **
%
%   'TIMES'
%           A vector of times for which the analysis should be performed.  In the above example, the data
%       spans 1000 ms, so 'times' might be 100:50:800.  This would compute the Fano factor every 50 ms starting
%       100 ms into the data and ending 200 ms before the data end.
%           You almost certainly DONT want to feed in every time (e.g. 1:1000 would be a bad choice).  This is
%       because 1) things will be slow, and 2) it becomes harder to match the distribution of mean counts as the
%       number of times increases.  As a result, more data will have to be excluded to manage the match.  Just 
%       ask for the times that are critical for your analysis, with a reasonable spacing (probably 20-50 ms 
%       between times is reasonable).
%           Also, keep in mind that the count window is centered on the times you ask for.  For example, the
%       default 'boxWidth' is 80 ms, which means the first entry in 'times' must be >40 ms.
%
%   'PARAMS'
%           This input structure is optional.  If you don't supply it, default values will be used.  You can
%       input params with some fields but not others (the unsupplied fields revert to the defaults).  The
%       fields are:
%           'boxWidth' is the width of the sliding window in which the count is made (default is 80 ms)
%           'matchReps' should be zero if you do not wish the firing rate distributions to 
%               be equalized across the times. When non-zero, that is the number of times 
%               the downselection is done (different random seed each time). (default is 10).
%           'binSpacing' determines the scale on which we try and match the firing rates.  It
%               is the bin width used to compute the distribution of spike counts. (default is 0.25 spikes).
%           'alignTime' allows you to specify the time that the neural responses are aligned to (e.g. target
%               onset). The 'times' field in the output will be with respect to that time.  This means that
%               you don't have to keep track of the alignment time for the purposes of subsequent plots / analyses.
%               (default is 0).
%           'initRand' If set to 1, 'rand' will be initialized. If set to 0, 'rand' is not initialized.  This
%               matters because 'rand' is used when downselecting data (deciding which points to ignore).
%               If you want the result to be identical every time 'VarVsMean' is called, don't set to 0.
%               So long as 'matchReps' isn't too small though, results will be very similar regardless
%               of whether rand is initialized. (default is 1).
%           'weightedRegression'  If set to 1, the regression takes into account the different sampling
%               error for the different points (lower trial counts and higher means result in a
%               larger sampling error in our measurement of the variance). Default is 1. 0 means not weighted.
%           'includeVarMinusMean'  If set to 1, 'Result' will contain fields that report the variance
%               minus the mean. (default is 0).
%    NOTE: whether you override the defaults or not, the function informs you of the values used on the command line.
% 
% OUTPUT: 'Result' has the following fields:
%        FanoFactor: The Fano Factor for each time (after downselection to match distributions across times)
%        Fano_95CIs: The 95% Confidence intervals on the Fano Factor
%       scatterData: contains the data for the variance versus mean scatterplot
%     FanoFactorAll: Same as 'FanoFactor' but for all datapoints (no downselection or matching)
%     FanoAll_95CIs: 95% CIs on the above
%    scatterDataAll: Scatterplot data for the above
%    meanRateSelect: The mean rate (across all conditions/neurons) after downselection (should be fairly constant)
%       meanRateAll: The mean rate with no downselection (i.e. the population average) (this and above in spikes/s) 
%          distData: The distributions of firing rate counts (raw and downselected)
%             times: The times that the analysis was performed, including offset if 'alignTime' was nonzero       
%                           Those below are produced only if 'includeVarMinusMean' == 1
%      varMinusMean: The variance minus the mean (avgd across all points, after mean-matching)
%         VMM_95CIs: Corresponding 95% CIs
%            VMMall: The 'raw' variance minus the mean (no mean-matching)
%      VMMall_95CIs: Corresponding 95% CIs
%  
function Result = VarVsMean(data, times, varargin)


% SET DEFAULTS AND OVERIDE IF A "PARAMS" STRUCTURE WAS PASSED
boxWidth = 80;  % width of the sliding window in which the counts are made
matchReps = 10; % number of random choices regarding which points to throw away when matching distributions
binSpacing = 0.25; % bin width when computing distributions of mean counts
alignTime = 0; % time of event that data are aligned to (in the output structure, times will be expressed relative to this)
initRand = 1;
weightedRegression = 1;
includeVarMinusMean = 0;
Pfields = {'boxWidth', 'matchReps', 'binSpacing', 'alignTime', 'initRand', 'weightedRegression', 'includeVarMinusMean'};
for i = 1:length(Pfields) % if a params structure was provided as an input, change the requested fields
    if ~isempty(varargin)&&isfield(varargin{1}, Pfields{i}), eval(sprintf('%s = varargin{1}.(Pfields{%d});', Pfields{i}, i)); end
    fprintf('%s = %g; ', Pfields{i}, eval(Pfields{i}) );
    if i==4, fprintf('\n'); end
end; fprintf('\n');
if initRand, rand('state',0); end  % initialize random number generator (done by default, can be overridden)
% warn if there is an unrecognized field in the input parameter structure
if ~isempty(varargin)  
    fnames = fieldnames(varargin{1});
    for i = 1:length(fnames)
        recognized = max(strcmp(fnames{i},Pfields));
        if recognized == 0, fprintf('fieldname %s not recognized\n',fnames{i}); end
    end
end
weightingEpsilon = 1 * boxWidth/1000; % past approx < 1 spike/s, weights don't increase much anymore
if ~islogical(data(1).spikes), disp('the spikes field should be of type logical'); end


% DO CONVOLUTION WITH SLIDING WINDOW.
% COMPUTE THE MEAN AND VARIANCE OF THE COUNT FOR EACH TIME & EACH NEURON/CONDITION  
for t = 1:length(times)
    scatterDataAll(t).time = times(t)-alignTime;
    scatterDataAll(t).var = zeros(length(data),1);  %initialize
    scatterDataAll(t).mn = zeros(length(data),1);   %initialize
end
maxRate = 0;  % used to keep track of max rate across all times / conds
trialCount = zeros(length(data),1);  % initialize for speed (used for weighted regression)
for cond = 1:length(data) 
    if isempty(data(cond).spikes), disp('ERROR: the field spikes is an empty matrix.'); end
    if size(data(cond).spikes, 1) < 2, disp('ERROR: in at least one case the field spikes has 1 or fewer trials.'); end
    Tstart = times - floor(boxWidth/2) + 1;
    Tend = times + ceil(boxWidth/2) + 1;
    if Tend(end)+1 > size(data(cond).spikes,2)
        disp('ERROR, BOX ENDS LATER THAN DATA FOR AT LEAST ONE TIME');
        disp('make sure there is >= (boxWidth/2 + 2) ms of data beyond the last requested time.');
        Result = []; return;
    end
    
    csum = cumsum(single(data(cond).spikes(:,Tstart(1):Tend(end)+1)), 2); % casting as single makes it faster
    count = csum(:,Tend - Tstart(1)+1) - csum(:,Tstart - Tstart(1)+1);

    varTemp = var(count);  % var and mean are taken for all times at once (to save time)
    mnTemp = mean(count);
    for t = 1:length(times)
        scatterDataAll(t).var(cond) = varTemp(t);
        scatterDataAll(t).mn(cond) = mnTemp(t);
    end
    maxRate = max(maxRate, max( mnTemp ) );
    
    trialCount(cond) = size(data(cond).spikes,1); % (used for weighted regression)
end

% COMPUTE FIRING RATE DISTRIBUTIONS AT EACH TIME TO BE TESTED
% This is necessary even if we aren't downselecting, to provide the user with bin and dist info
meanRates = zeros(length(times),2);  % initialize to save time
bins = 0:binSpacing:maxRate+binSpacing;  % bins that define the precision of the match
for t = 1:length(times)
    meanRates(t,1) = 1000 / boxWidth * mean(scatterDataAll(t).mn); % convert to spikes/s (ONLY for the rate)
    stdOfDeltaRate(t,1) = 1000 / boxWidth * std(scatterDataAll(t).mn - scatterDataAll(1).mn); % standard deviation of means
    [distData(t).counts, distData(t).whichBin] = histc(scatterDataAll(t).mn,bins);
    if distData(t).counts(end) > 0, disp('ERROR: last bin contained more than one datapoint'); end; % the last element is the number outside the bin range (should always be zero)
    distData(t).binEdges = bins';  % just the bins themselves (so the user can use them for plotting etc)
end    
targetCount = min( [distData.counts], [], 2);  % takes min count, across the measurement times, for each bin.



% COMPUTE SLOPE VERSUS TIME FOR ENTIRE DATA SET (no distribution matching yet)
for t = 1:length(times)
    if weightedRegression % the variance of the estimate of the variance (whew!) scales as the square of the mean, and inversely with trial count
        regWeights = trialCount ./ (scatterDataAll(t).mn + weightingEpsilon) .^ 2;
    else
        regWeights = ones(size(scatterDataAll(t).mn));
    end
    [B, stdB] = lscov(scatterDataAll(t).mn, scatterDataAll(t).var, regWeights);
    Bint = [B-2*stdB, B+2*stdB];
    slopesAll(t,1) = B;
    slopesAll_95CIs(t,1:2) = Bint;
    
    if includeVarMinusMean  % if asked by user (default is not)
        meanTemp = mean(scatterDataAll(t).var - scatterDataAll(t).mn);
        semTemp = std(scatterDataAll(t).var - scatterDataAll(t).mn) / sqrt(length(scatterDataAll(t).var));
        VMMall(t,1) = meanTemp;
        VMMall_95CIs(t,1:2) = meanTemp + [-2, 2]*semTemp;
    end
end



% NOW DO THE DISTRIBUTION MATCHING
if matchReps > 0      
    scatterDataSelect = scatterDataAll;  % this will be downselected
    
    sumSlopes = zeros(length(times),1);  % used to compute mean slope across downselection draws
    sumCIs = zeros(length(times),2);     % ditto for the CI on that slope
    
    if includeVarMinusMean
        sumDiffs = zeros(length(times),1); % initialize for speed (parallel with sumSlopes above)
        sumDiffCIs = zeros(length(times),2);
    end
        
    for rep = 1:matchReps   % distribution matching gets done 'matchReps' times
        for t = 1:length(times)
            toKeep = [];
            for b = 1:length(bins)-1  % go through bins and figure out which datapoints to keep
                thisBin = find( distData(t).whichBin == b );  % indices for means in this bin            
                if length(thisBin) < targetCount(b), disp('ERROR: target count not right'); end            
                thisBin = thisBin(randperm(length(thisBin)));  % shuffle before cutting off the end.
                if targetCount(b)~=0
                    toKeep = [toKeep; thisBin(1:targetCount(b))];  % keep at most all 
                end
                distData(t).countsSelect(b,1) = length(thisBin(1:targetCount(b)));
            end
            distData(t).countsSelect(length(bins),1) = 0;  % All dists should end with a zero
            
            if length(toKeep) < 5
                disp(' ');
                disp('PROBLEM IN VarVsMean !!!!!!!!!!!!!!!!');
                disp('little or no data survived mean matching');
                disp('either params.binSpacing is too small, or there is no overlap for some times');
                disp('you should fix this, or give up on mean matching and set matchReps = 0');
                disp('returning with no output arguments');
                return;
            end
            
            mnsThisRep = scatterDataAll(t).mn(toKeep);
            varsThisRep = scatterDataAll(t).var(toKeep);

            if weightedRegression  % default is weighted
                regWeights = trialCount(toKeep) ./ (mnsThisRep + weightingEpsilon).^2;
            else
                regWeights = ones(size(mnsThisRep));
            end
            [B, stdB] = lscov(mnsThisRep,varsThisRep, regWeights);
            Bint = [B-2*stdB, B+2*stdB]; 
            
            sumSlopes(t,1) = sumSlopes(t,1) + B;
            sumCIs(t,:) = sumCIs(t,:) + Bint;
                       
            if rep == 1 % the returned scatterpoints are from the first rep
                scatterDataSelect(t).mn = mnsThisRep;
                scatterDataSelect(t).var = varsThisRep;
            end
            
            meanRates(t,2) = meanRates(t,2) + 1000/boxWidth*mean(mnsThisRep)/matchReps; % mean over all reps
            
            % Variance MINUS Mean
            if includeVarMinusMean  % if asked by user (default is not)
                meanTemp = mean(varsThisRep - mnsThisRep);
                semTemp = std(varsThisRep - mnsThisRep) / sqrt(length(varsThisRep));
                sumDiffs(t,1) = sumDiffs(t,1) + meanTemp;
                sumDiffCIs(t,:) = sumDiffCIs(t,:) + meanTemp + semTemp*[-2, 2];
            end
        end
    end
    
    slopes = sumSlopes / matchReps;  % computes the mean from the sum ('matchReps' repeats were used)
    slopes_95CIs = sumCIs / matchReps;
    
    if includeVarMinusMean  % if the user also wants the variance minus the mean
        diffs = sumDiffs / matchReps;
        diffs_95CIs = sumDiffCIs / matchReps;
    end
    
    fprintf('%d percent of datapoints survived matching\n', round(100*sum(distData(1).countsSelect) / sum(distData(1).counts)));        
end



% DONE WITH COMPUTATIONS
% MAKE OUTPUT STRUCTURE: 'Result'
if matchReps > 0
    Result.FanoFactor = slopes; % slopes (distribution matched)
    Result.Fano_95CIs = slopes_95CIs; % 95% CI's on those slopes
    Result.scatterData = scatterDataSelect';  % data for scatterplot (distribution matched)
end

Result.FanoFactorAll = slopesAll;  % slopes (not distribution matched)
Result.FanoAll_95CIs = slopesAll_95CIs; % 95% CI's on those slopes
Result.scatterDataAll = scatterDataAll';  % data for scatterplot (not distribution matched)

if matchReps > 0        
    Result.meanRateSelect = meanRates(:,2); % the 'mean mean-rate' across time, after matching (should change little)
end                             % mean rates are in spikes/s (not spike count, like everything else)
Result.meanRateAll = meanRates(:,1); % the 'mean mean-rate' across time, no matching (just the pop mean)
Result.stdOfDeltaRate = stdOfDeltaRate(:,1);
distData = rmfield(distData,'whichBin'); %doesn't need to be returned to user
Result.distData = distData';  % distributions of spike counts for the different times
Result.times = times' - alignTime;  % just the times that the user asked for, stored here for convenience

if includeVarMinusMean  % if user asks
    Result.varMinusMean = diffs;
    Result.VMM_95CIs = diffs_95CIs;
    
    Result.VMMall = VMMall;
    Result.VMMall_95CIs = VMMall_95CIs;
end
    


























    