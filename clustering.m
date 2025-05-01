clear; clc; close all;

load("data_all.mat");

M = 64;
numClasses = 10;

% Empty vectors, allocating memory
clusters = zeros(M*10,vec_size); 
clusterClass = zeros(M*10,1);

tic
% Iterate thorugh all the digits
for i = 0:9
    train_i = trainv(trainlab == i,:);  % only uses training vectorwith label that equals current class i
    [~,C] = kmeans(train_i,M);          % making C with cluster centers

    rowStart = M * i + 1;
    rowEnd = M * (i + 1);

    clusters(rowStart:rowEnd, :) = C;
    clusterClass(rowStart:rowEnd) = i*ones(M,1);

    fprintf('Class %d clustered\n',i);
end
toc

save('clusters.mat', 'clusters', 'clusterClass');

