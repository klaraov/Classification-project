clear all
close all

%% Task 1.2a) Iris classification with alpha=0.01, drop features with greatest overlap between classes

setClass1 = load('class_1');
setClass2 = load('class_2');
setClass3 = load('class_3');
disp('Taken away: Sepal Width, Sepal Length and Petal Length')

setClass1(:,[1,2,3]) = [];
setClass2(:,[1,2,3]) = [];
setClass3(:,[1,2,3]) = [];

numClasses = 3;
numFeatures = 1;
numTrainPerClass = 30;
numTestPerClass  = 20;

% Prepare training and testing datasets
trainData = [setClass1(1:numTrainPerClass,:).', setClass2(1:numTrainPerClass,:).', setClass3(1:numTrainPerClass,:).'];
testData  = [setClass1(numTrainPerClass+1:end,:).', setClass2(numTrainPerClass+1:end,:).', setClass3(numTrainPerClass+1:end,:).'];

% One-hot encoded labels
label1 = [1;0;0]; label2 = [0;1;0]; label3 = [0;0;1];
targetLabels = [repmat(label1,1,numTrainPerClass), repmat(label2,1,numTrainPerClass), repmat(label3,1,numTrainPerClass)];

% Training configuration
learningRate = 0.01;
maxIters = 3000;

% Weight and bias initialization
weights = zeros(numClasses, numFeatures);
bias = zeros(numClasses, 1);
lossCurve = zeros(1, maxIters);

% Helper functions
activationFn = @(z) 1./(1+exp(-z));
lossFn = @(output,target) 0.5*(output-target)'*(output-target);
gradLossFn = @(output,target,input) ((output-target).*output.*(1-output)) * input.';

% Training loop
numTrainSamples = size(trainData,2);
for iter = 1:maxIters
    gradAccum = zeros(numClasses, numFeatures+1);
    lossSum = 0;
    for idxSample = 1:numTrainSamples
        inputVec = [trainData(:,idxSample); 1];
        trueLabel = targetLabels(:,idxSample);
        zOut = weights*trainData(:,idxSample) + bias;
        prediction = activationFn(zOut);
        gradStep = gradLossFn(prediction, trueLabel, inputVec);
        gradAccum = gradAccum + gradStep;
        lossSum = lossSum + lossFn(prediction, trueLabel);
    end
    % Update weights and biases
    weights = weights - learningRate * gradAccum(:,1:numFeatures);
    bias = bias - learningRate * gradAccum(:,numFeatures+1);
    lossCurve(iter) = lossSum;
end

% Plot loss curve
figure;
plot(1:maxIters, lossCurve, 'LineWidth',1.5);
xlabel('Iteration'); ylabel('Total Training MSE');
title('\alpha = 0.01, dropped features 1,2,3');
grid on;

% Confusion matrix and error rate: Training set
confTrain = zeros(numClasses);
for idxSample = 1:numTrainSamples
    zOut = weights*trainData(:,idxSample) + bias;
    [~, predictedClass] = max(activationFn(zOut));
    trueClass = ceil(idxSample/numTrainPerClass);
    confTrain(trueClass, predictedClass) = confTrain(trueClass, predictedClass) + 1;
end
errTrain = 1 - trace(confTrain)/(numTrainSamples);

disp('Confusion matrix (train):'); disp(confTrain);
disp(['Error rate (train): ', num2str(errTrain)]);

% Confusion matrix and error rate: Test set
numTestSamples = size(testData,2);
confTest = zeros(numClasses);
for idxSample = 1:numTestSamples
    zOut = weights*testData(:,idxSample) + bias;
    [~, predictedClass] = max(activationFn(zOut));
    trueClass = ceil(idxSample/numTestPerClass);
    confTest(trueClass, predictedClass) = confTest(trueClass, predictedClass) + 1;
end
errTest = 1 - trace(confTest)/(numTestSamples);

disp('Confusion matrix (test):'); disp(confTest);
disp(['Error rate (test): ', num2str(errTest)]);

% Training confusion chart
figure;
cm_train = confusionchart(confTrain, ...
    'RowSummary','row-normalized', ...
    'ColumnSummary','column-normalized');
cm_train.Title = 'Confusion Matrix - Training Set';
cm_train.FontSize = 14;
cm_train.XLabel = 'Predicted Class';
cm_train.YLabel = 'True Class';

% Test confusion chart
figure;
cm_test = confusionchart(confTest, ...
    'RowSummary','row-normalized', ...
    'ColumnSummary','column-normalized');
cm_test.Title = 'Confusion Matrix - Test Set';
cm_test.FontSize = 14



