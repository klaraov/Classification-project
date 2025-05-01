clear; clc; close all;

% Load data
load('data_all.mat');
load('nn_predictions.mat');

% Get all correct and incorrect classification indices
[correct_indices, incorrect_indices] = ...
    get_classification_indices(testlab, nn_predictions, testv);

% Random chosen pictures from correct and incorrect indices list for display
sample_incorrect = [1040, 1466, 9983, 4824];
sample_correct = [1235,    7, 9919, 9470];

image_matrix = zeros(28,28);

% Display wrongly classified
figure(1)
sgtitle("Wrongly Classified Pictures", 'FontSize', 20)

for i = 1:numel(sample_incorrect)
    idx = sample_incorrect(i);
    image_matrix(:) = testv(idx, :);
    subplot(2,2,i);
    image(rot90(flip(image_matrix), 3));
    
    actual_label = testlab(idx);
    [~, pred_idx] = max(nn_predictions(:, idx));
    predicted_label = pred_idx - 1;
    
    title(sprintf('Pred: %d    True: %d', predicted_label, actual_label), ...
        'FontSize', 15, 'FontWeight', 'bold');
end

% Display correctly classified
figure(2)
sgtitle("Correctly Classified Pictures", 'FontSize', 20)

for i = 1:numel(sample_correct)
    idx = sample_correct(i);
    image_matrix(:) = testv(idx, :);
    subplot(2,2,i);
    image(rot90(flip(image_matrix), 3));
    
    actual_label = testlab(idx);
    [~, pred_idx] = max(nn_predictions(:, idx));
    predicted_label = pred_idx - 1;
    
    title(sprintf('Pred: %d    True: %d', predicted_label, actual_label), ...
        'FontSize', 15, 'FontWeight', 'bold');
end

function [correct_indices, incorrect_indices] = get_classification_indices( ...
    true_labels, prediction_probs, test_data)

    num_samples = size(test_data, 1);
    num_correct = 0;
    num_incorrect = 0;

    % Count how many correct vs incorrect
    for k = 1:num_samples
        [~, pred_idx] = max(prediction_probs(:, k));
        if true_labels(k) + 1 == pred_idx
            num_correct = num_correct + 1;
        else
            num_incorrect = num_incorrect + 1;
        end
    end

    % Allocate index arrays
    correct_indices = zeros(num_correct, 1);
    incorrect_indices = zeros(num_incorrect, 1);
    corr_counter = 1;
    incor_counter = 1;

    % Fill in the indices
    for k = 1:num_samples
        [~, pred_idx] = max(prediction_probs(:, k));
        if true_labels(k) + 1 == pred_idx
            correct_indices(corr_counter) = k;
            corr_counter = corr_counter + 1;
        else
            incorrect_indices(incor_counter) = k;
            incor_counter = incor_counter + 1;
        end
    end
end
