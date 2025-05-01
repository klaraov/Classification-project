clear; clc; close all;

load("data_all.mat");

num_test = size(testv, 1);

tic
% Call nearest neighbor classifier
[nn_predictions] = nn_classifier_func(trainv, testv, trainlab, 1000);

% Make one-hot known labels for the test set
knowns = zeros(10, size(nn_predictions, 2));
% For each test sample, index k, give its true digit
for k = 1:size(nn_predictions, 2)
    knowns(testlab(k) + 1, k) = 1;     % +1 to make it to MATLAB index
end
toc

% Confusion matrix
confusion_nn = knowns * nn_predictions';

% Plot confusion matrix 
figure(1)
plot_confusion_matrix(confusion_nn, 0:9, 'NN Classifier');

% Error rate 
error_rate = (num_test - trace(confusion_nn)) / num_test;
disp('Error rate for NN without clustering:')
disp(error_rate);

save nn_predictions nn_predictions

function predictions = nn_classifier_func(train_data, test_data, train_labels, chunk_size)
    num_test    = size(test_data, 1);                  % total number of test samples
    num_classes = 10;                                  % MNIST has digits 0-9
    predictions = zeros(num_classes, num_test);        % onehot prediction matrix
    num_chunks  = ceil(num_test / chunk_size);         % how many chunks we need
    
    % Loop through chunks
    for c = 1:num_chunks
        fprintf('Processing chunk %d of %d...\n', c, num_chunks);
        idx_start     = (c-1) * chunk_size + 1;
        idx_end       = min(c * chunk_size, num_test);
        current_chunk = test_data(idx_start:idx_end, :);

        % Compute Euclidean distances between each test sample and all training samples
        D = pdist2(train_data, current_chunk, 'euclidean');

        % For each test sample, find the nearest training sample
        [~, min_idx] = min(D, [], 1);
        
        % Convert nearest neighbor index to class labels and set one hot
        for j = 1:length(min_idx)
            class_idx = train_labels(min_idx(j)) + 1;  % MATLAB uses 1 indexing, not 0
            predictions(class_idx, idx_start + j - 1) = 1;
        end
    end
end
