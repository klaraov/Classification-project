clear; clc; close all;

load('data_all.mat');
load('clusters.mat');

num_test = size(testv, 1);
classes = 10;

tic
predictedNumbers = zeros(classes,num_test);
for i = 1:num_test
    D = dist(clusters, testv(i,:).');
    [~,index] = min(D);
    number = clusterClass(index);
    predictedNumbers(number+1,i) = 1;    
end
toc

knowns =  zeros(10, num_test);
for k = 1:num_test
    knowns(testlab(k)+1,k) = 1;
end

% Confusion Matrix
confClusterNN = knowns * predictedNumbers';
plotConfusionMatrix(confClusterNN, 0:9,'Clustered NN Classifier');

% Error rate 
errorRate = (num_test-trace(confClusterNN))/num_test;
disp('Error rate with NN with clustering:')
disp(errorRate);
