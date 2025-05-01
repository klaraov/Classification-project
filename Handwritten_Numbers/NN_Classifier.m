clear; clc; close all;

load("data_all.mat");

tic

% Call nearest neigbor classifier
[NN_pred] = NNClassifierFunc(trainv, testv, trainlab, 1000);

% Make one-hot known lables for the test set
knowns =  zeros(10, size(NN_pred,2));

% For each test sample, index k, give its true digit
for k = 1:size(NN_pred,2)
    knowns(testlab(k)+1,k) = 1;     % + 1 to make it to MATLAB index
end

toc

% Confusion matrix
confusionNN = knowns * NN_pred';

% Plot confusion matrix 
figure(1)
plotConfusionMatrix(confusionNN, 0:9,'NN Classifier');

% Error Rate 
errorRate = (num_test-trace(confusionNN))/num_test;
disp('Error rate for NN without clustering:')
disp(errorRate);

save NN_pred

function [pred] = NNClassifierFunc(trainv, testv, trainlab, chunkSize)

    num_test = size(testv, 1);                  % Total number of test samples
    numClasses = 10;                            % MNIST has digits 0-9
    pred = zeros(numClasses, num_test);         % one-hot prediction matrix
    numChunks = ceil(num_test / chunkSize);     % How many chuncks we need
    
    % Loop thorugh chuncks
    for c = 1:numChunks
        fprintf('Processing chunk %d of %d...\n', c, numChunks);
        idx_start = (c-1) * chunkSize + 1;
        idx_end = min(c * chunkSize, num_test);
        currentChunk = testv(idx_start:idx_end, :);

        % Compute Euclidean distances between each test sample and all training samples
        D = pdist2(trainv, currentChunk, 'euclidean');

        % For each test sample, find the nearest training sample
        [~, minIdx] = min(D, [], 1);  % find index of nearest neighbor in training set
        
        % Convert nearest neighbor index to class labels and set one‐hot
        for j = 1:length(minIdx)
            classIdx = trainlab(minIdx(j)) + 1;  % class labels are 0–9, adjust to 1–10. MATLAB is not 0 indexed
            pred(classIdx, idx_start + j - 1) = 1;
        end
    end
end
