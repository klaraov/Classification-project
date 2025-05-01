clear; clc; 
close all;

load('data_all.mat');
load('clusters.mat');

num_test = size(testv, 1);
num_classes = 10;

tic
predicted_matrix = zeros(num_classes, num_test);
for i = 1:num_test
    distances = dist(clusters, testv(i, :).');
    [~, nearest_idx] = min(distances);
    assigned_class  = cluster_class(nearest_idx);
    predicted_matrix(assigned_class + 1, i) = 1;    
end
toc

known_matrix = zeros(num_classes, num_test);
for k = 1:num_test
    known_matrix(testlab(k) + 1, k) = 1;
end

% Confusion Matrix
confusion_matrix = known_matrix * predicted_matrix';
plot_confusion_matrix(confusion_matrix, 0:9, 'Clustered NN Classifier');

% Error rate 
error_rate = (num_test - trace(confusion_matrix)) / num_test;
disp('Error rate with NN with clustering:')
disp(error_rate);
