clear; clc; close all;

load('data_all.mat');

num_clusters_per_class = 64;     % number of clusters per digit class
num_classes = 10;

% Empty vectors, allocating memory
clusters = zeros(num_clusters_per_class * num_classes, vec_size);
cluster_class = zeros(num_clusters_per_class * num_classes, 1);

tic
% Iterate through all the digits
for digit = 0:num_classes-1
    train_subset = trainv(trainlab == digit, :); % only uses training vectors whose label equals current class digit
    [~, cluster_centers] = kmeans(train_subset, num_clusters_per_class); % finding cluster centers

    row_start = num_clusters_per_class * digit + 1;
    row_end = num_clusters_per_class * (digit + 1);

    clusters(row_start:row_end, :) = cluster_centers;
    cluster_class(row_start:row_end) = digit * ones(num_clusters_per_class, 1);

    fprintf('Class %d clustered\n', digit);
end
toc

save('clusters.mat', 'clusters', 'cluster_class');
