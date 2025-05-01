clear all
close all

% Choose the first 30 samples for training and the last 20 samples for testing.
learningRates = [0.1, 0.05, 0.01];
numRates = numel(learningRates);
maxIters = 3000;
lossHistory = zeros(numRates, maxIters);

% Initial settings
numClasses = 3;
numFeatures = 4;

% Load datasets
setClass1 = load('class_1');
setClass2 = load('class_2');
setClass3 = load('class_3');

% Training and testing data size
numTrainPerClass = 30;  
totalTrain = numTrainPerClass * numClasses;
numTestPerClass = 20;   
totalTest = numTestPerClass * numClasses;

% Compose train/test sets
trainData = [setClass1(1:numTrainPerClass,:).', setClass2(1:numTrainPerClass,:).', setClass3(1:numTrainPerClass,:).'];
testData  = [setClass1(numTrainPerClass+1:end,:).', setClass2(numTrainPerClass+1:end,:).', setClass3(numTrainPerClass+1:end,:).'];

% Define onehot targets
label1 = [1;0;0]; label2 = [0;1;0]; label3 = [0;0;1];
targetLabels = [repmat(label1,1,numTrainPerClass), repmat(label2,1,numTrainPerClass), repmat(label3,1,numTrainPerClass)];

activationFn = @(z) 1./(1+exp(-z));
lossFn = @(output,target) 0.5*(output-target)'*(output-target);
gradLossFn = @(output,target,input) ((output-target).*output.*(1-output))*input.';

%% Training using multiple learning rates
for idxRate = 1:numRates
    rate = learningRates(idxRate);

    % Initialize weights and biases
    weights = zeros(numClasses, numFeatures);
    bias = zeros(numClasses,1);

    for iter = 1:maxIters
        gradAccum = zeros(numClasses, numFeatures+1);
        totalLoss = 0;

        for idxSample = 1:size(trainData,2)
            inputVec = [trainData(:,idxSample); 1];
            trueLabel = targetLabels(:,idxSample);
            zOut = weights*trainData(:,idxSample) + bias;
            prediction = activationFn(zOut);

            gradStep = gradLossFn(prediction, trueLabel, inputVec);
            gradAccum = gradAccum + gradStep;
            totalLoss = totalLoss + lossFn(prediction, trueLabel);
        end

        % Update model
        weights = weights - rate * gradAccum(:,1:numFeatures);
        bias = bias - rate * gradAccum(:,numFeatures+1);

        lossHistory(idxRate, iter) = totalLoss;
    end
end

%% Plot loss curves
figure;
hold on;
epochList = 1:maxIters;
for idxRate = 1:numRates
    plot(epochList, lossHistory(idxRate,:), 'LineWidth',1.5);
end
xlabel('Iteration');
ylabel('Total Training MSE');
title('MSE for different \alpha');
legend(arrayfun(@(a) sprintf('\\alpha=%.3f',a), learningRates,'UniformOutput',false),'Location','northeast');
grid on;
set(gca, 'FontSize', 18);
hold off;

%% Confusion matrix and error rates
trainConfMat = zeros(numClasses);
for idxSample = 1:totalTrain
    inputVec = [trainData(:,idxSample);1];
    trueClass = floor((idxSample-1)/numTrainPerClass)+1;
    output = weights*inputVec(1:end-1) + bias;
    [~,predictedClass] = max(activationFn(output));
    trainConfMat(trueClass, predictedClass) = trainConfMat(trueClass, predictedClass) + 1;
end
trainError = 1 - trace(trainConfMat)/totalTrain;

disp('Error rate - Training:'); disp(trainError);
disp('Confusion matrix - Training:'); disp(trainConfMat);

testConfMat = zeros(numClasses);
for idxSample = 1:totalTest
    inputVec = [testData(:,idxSample);1];
    trueClass = floor((idxSample-1)/numTestPerClass)+1;
    output = weights*inputVec(1:end-1) + bias;
    [~,predictedClass] = max(activationFn(output));
    testConfMat(trueClass, predictedClass) = testConfMat(trueClass, predictedClass) + 1;
end
testError = 1 - trace(testConfMat)/totalTest;

disp('Error rate - Testing:'); disp(testError);
disp('Confusion matrix - Testing:'); disp(testConfMat);
