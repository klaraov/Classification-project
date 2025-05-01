clear; clc; close all;

load('data_all.mat');
load('clusters.mat');

K = 7; % compare to the 7 nearest neighbors
NumClasses = 10;
predictedNumbers = zeros(NumClasses, num_test); % allocate

tic
for i = 1:num_test
    idx = knnsearch(clusters, testv(i,:), 'K', K); 
    countNumbers = zeros(NumClasses,1);
    for j = 1:K
        number = clusterClass(idx(j));
        countNumbers(number+1) = countNumbers(number+1) + 1;
    end
    
    [~,index] = max(countNumbers);
    predictedNumbers(index,i) = 1;   
end
toc

knowns =  zeros(10, num_test);
for k = 1:num_test
    knowns(testlab(k)+1,k) = 1;
end

% Confusion matrix
KNNconfusion = knowns * predictedNumbers';
plotConfusionMatrix(KNNconfusion, 0:9,'Clustered KNN Classifier (K=7)');

% Error rate
errorRate = (num_test - trace(KNNconfusion)) / num_test;
disp('Error rate with KNN with clustering (K=7):')
disp(errorRate)

