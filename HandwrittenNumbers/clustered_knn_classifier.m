clear; clc; close all;

load('data_all.mat');
load('clusters.mat');

k = 7; % compare to the 7 nearest neighbors
num_classes = 10;
num_test = size(testv, 1);

predicted_matrix = zeros(num_classes, num_test); % allocate

tic
for i = 1:num_test
    idx = knnsearch(clusters, testv(i,:), 'K', k); 
    vote_counts = zeros(num_classes,1);
    for j = 1:k
        number = cluster_class(idx(j));
        vote_counts(number+1) = vote_counts(number+1) + 1;
    end
    
    [~, max_idx] = max(vote_counts);
    predicted_matrix(max_idx, i) = 1;   
end
toc

known_matrix = zeros(num_classes, num_test);
for m = 1:num_test
    known_matrix(testlab(m)+1, m) = 1;
end

% Confusion matrix
knn_confusion = known_matrix * predicted_matrix';
plot_confusion_matrix(knn_confusion, 0:9,'Clustered KNN Classifier (K=7)');

% Error rate
error_rate = (num_test - trace(knn_confusion)) / num_test;
disp('Error rate with KNN with clustering (K=7):')
disp(error_rate)


