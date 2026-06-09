%% LRMGC_main
clc
clear
close all
clear memory

%%
addpath('.\function')
addpath('.\dataset')

%cells*features
dataset = {'Ting', 'islet'}

for ii = 1:length(dataset)
    ii
    load(dataset{ii})
    
    %%
    X = in_X';
    nnClass = length(unique(true_labs));
    gnd = true_labs;
    lambda1 = 0.9;
    lambda2 = 0.05;
    p = 0.6;
    maxIter = 30;
    tic
    
    %%
    [Z,C,error] = LRMGC_opt(X, lambda1, lambda2, p, maxIter);
    W = postprocessor(Z);
    similarity = (abs(W)+abs(W'))/2;
    [result_label, kerNS] = SpectralClustering(similarity, nnClass);
    %     result = [ACC NMI Fscore Precision ARI Purity Recall Entropy];
    result = Clustering8Measure(gnd, result_label);
    
    %%
    time(ii) = toc;
    ACC(ii) = ACC_ClusteringMeasure(gnd, result_label);
    ARI(ii) = Contingency_ARI_newused(gnd,result_label);
    NMI(ii) = Cal_NMI_newused(gnd, result_label);
    fprintf('ACC is %f\n',ACC(ii))
    fprintf('ARI is %f\n',ARI(ii))
    fprintf('NMI is %f\n',NMI(ii))
end

