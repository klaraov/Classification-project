clear all
close all

%% Task 1.1a)
% Choose the first 30 samples for training and the last 20 samples for testing.
% alphas    = [0.1, 0.05, 0.01 ,0.005,0.001];
alphas    = [0.1, 0.05, 0.01];
numAlphas = numel(alphas);
NumIterations = 3000;     
MSEs_all  = zeros(numAlphas, NumIterations);

% Defining initial variables
C = 3;                          % Number of classes
D = 4;                          % Number of features per sample
% Load data
dataClass1 = load('class_1');
dataClass2 = load('class_2');
dataClass3 = load('class_3');

% Training and test indices
NumTrainC = 30;  NumTrain  = NumTrainC * C;
NumTestC  = 20;  NumTest   = NumTestC  * C;

% Prepare train and test sets
%first 30 for training
trainSet = [dataClass1(1:NumTrainC,:).', dataClass2(1:NumTrainC,:).', dataClass3(1:NumTrainC,:).'];
testSet  = [dataClass1(NumTrainC+1:end,:).', dataClass2(NumTrainC+1:end,:).', dataClass3(NumTrainC+1:end,:).'];


%Last 30 for training.
%trainSet = [dataClass1(NumTestC+1:end,:).', dataClass2(NumTestC+1:end,:).', dataClass3(NumTestC+1:end,:).'];
%testSet  = [dataClass1(1:NumTestC,:).', dataClass2(1:NumTestC,:).', dataClass3(1:NumTestC,:).'];

% Targets: one-hot for each class in training set
t1 = [1;0;0]; t2 = [0;1;0]; t3 = [0;0;1];
targets = [repmat(t1,1,NumTrainC), repmat(t2,1,NumTrainC), repmat(t3,1,NumTrainC)];

% Helper functions
sigmoid    = @(z) 1./(1+exp(-z));
MSE_fn     = @(g,t) 0.5*(g-t)'*(g-t);
gradMSE_fn = @(g,t,x) ((g-t).*g.*(1-g))*x.';

%% --- Training over multiple alphas ---
for iAlpha = 1:numAlphas
    alpha = alphas(iAlpha);

    % Initialize weights (including bias)
    W  = zeros(C, D);
    b  = zeros(C,1);

    % Gradient-descent for NumIterations
    for m = 1:NumIterations
        Gtot = zeros(C, D+1);   % gradient accumulator (last col = bias)
        MSE  = 0;
        
        for k = 1:size(trainSet,2)
            xk = [trainSet(:,k); 1];     % append 1 for bias
            tk = targets(:,k);
            zk = W*trainSet(:,k) + b;
            gk = sigmoid(zk);

            Gk = gradMSE_fn(gk, tk, xk);  % size C x (D+1)
            Gtot = Gtot + Gk;
            MSE  = MSE + MSE_fn(gk, tk);
        end

        % Update weights and bias
        W = W - alpha * Gtot(:,1:D);
        b = b - alpha * Gtot(:,D+1);

        % Store MSE
        MSEs_all(iAlpha, m) = MSE;
    end
end

%% --- Plot MSE curves for all alphas ---
figure;
hold on;
iters = 1:NumIterations;
for iAlpha = 1:numAlphas
    plot(iters, MSEs_all(iAlpha,:), 'LineWidth',1.5);
end
xlabel('Iteration');
ylabel('Total Training MSE');
title('MSE for different \alpha');
legend(arrayfun(@(a) sprintf('\\alpha=%.3f',a), alphas,'UniformOutput',false),'Location','northeast');
grid on;
set(gca, 'FontSize', 18);
hold off;

%% Task 1.1c) - Confusion and error rates (using final W,b of last alpha)
% (You can choose to rerun for a specific alpha or store Ws/bs per alpha.)
confusionTrain = zeros(C);
for k = 1:NumTrain
    xk = [trainSet(:,k);1];
    trueClass = floor((k-1)/NumTrainC)+1;
    zk = W*xk(1:end-1) + b;
    [~,predClass] = max(sigmoid(zk));
    confusionTrain(trueClass, predClass) = confusionTrain(trueClass, predClass) + 1;
end
errorRateTrain = 1 - trace(confusionTrain)/NumTrain;

disp('Error rate - Training:'); disp(errorRateTrain);
disp('Confusion matrix - Training:'); disp(confusionTrain);

confusionTest = zeros(C);
for k = 1:NumTest
    xk = [testSet(:,k);1];
    trueClass = floor((k-1)/NumTestC)+1;
    zk = W*xk(1:end-1) + b;
    [~,predClass] = max(sigmoid(zk));
    confusionTest(trueClass, predClass) = confusionTest(trueClass, predClass) + 1;
end
errorRateTest = 1 - trace(confusionTest)/NumTest;

disp('Error rate - Testing:'); disp(errorRateTest);
disp('Confusion matrix - Testing:'); disp(confusionTest);
