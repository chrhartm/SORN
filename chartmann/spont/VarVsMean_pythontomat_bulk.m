function [ result ] = VarVsMean_pythontomat_bulk( data )
[words,trials,steps,neurons] = size(data);

boxsize = 5;

params.boxWidth = boxsize;%1
params.matchReps = 10;%for mean matching (0->no mean matching)
params.binSpacing= 0.25;%binsize for matching (big bins: mean fluctuates, small bins: data thrown away)
params.weightedRegression = 1;

for word=1:words
    for n=1:neurons  
        tmp = squeeze(data(word,:,:,n))==1;
        data_new((word-1)*neurons+n).spikes = tmp;
    end
end

tmpresult = VarVsMean(data_new,ceil(boxsize/2):1:steps-ceil(boxsize/2)-2,params);
FFs = tmpresult.FanoFactor;
FFsAll = tmpresult.FanoFactorAll;
FFs95 = tmpresult.Fano_95CIs;
FFs95All = tmpresult.FanoAll_95CIs;
means = tmpresult.meanRateSelect;
meansAll = tmpresult.meanRateAll;

result = [FFs, FFs95, means, FFsAll, FFs95All, meansAll];
end
