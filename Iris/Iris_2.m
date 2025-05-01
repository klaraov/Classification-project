clear all
close all

%% Task 1.2a) Iris classification with alpha=0.01, drop features with greatest overlap between classes

dataClass1 = load('class_1');
dataClass2 = load('class_2');
dataClass3 = load('class_3');
disp('Taken away: Sepal Width, Sepal Length and Petal Length')

dataClass1(:,[1,2,3]) = [];
dataClass2(:,[1,2,3]) = [];
dataClass3(:,[1,2,3]) = [];

C = 3;                     % number of classes
D = 1;                     % remaining number of features
NumTrainC = 30;            % per-class train samples
NumTestC  = 20;            % per-class test samples

% Prepare train and test sets (each column = one sample)
trainSet = [dataClass1(1:NumTrainC,:).', dataClass2(1:NumTrainC,:).', dataClass3(1:NumTrainC,:).'];
testSet  = [dataClass1(NumTrainC+1:end,:).', dataClass2(NumTrainC+1:end,:).', dataClass3(NumTrainC+1:end,:).'];

% One-hot target vectors
t1 = [1;0;0]; t2 = [0;1;0]; t3 = [0;0;1];
targets = [repmat(t1,1,NumTrainC), repmat(t2,1,NumTrainC), repmat(t3,1,NumTrainC)];

% Hyperparameters
goodAlpha    = 0.01;       % fixed step size
epochs       = 3000;

% Initialization
W = zeros(C, D);
b = zeros(C, 1);
MSEs = zeros(1, epochs);

% Helpers
sigmoid    = @(z) 1./(1+exp(-z));
MSE_fn     = @(g,t) 0.5*(g-t)'*(g-t);
gradMSE_fn = @(g,t,x) ((g-t).*g.*(1-g)) * x.';

% Gradient descent training
totSamples = size(trainSet,2);
for m = 1:epochs
    Gtot = zeros(C, D+1);
    mseAcc = 0;
    for k = 1:totSamples
        xk = [trainSet(:,k); 1];
        tk = targets(:,k);
        zk = W*trainSet(:,k) + b;
        gk = sigmoid(zk);
        Gk = gradMSE_fn(gk, tk, xk);
        Gtot = Gtot + Gk;
        mseAcc = mseAcc + MSE_fn(gk, tk);
    end
    % Update
    W = W - goodAlpha * Gtot(:,1:D);
    b = b - goodAlpha * Gtot(:,D+1);
    MSEs(m) = mseAcc;
end

% Plot MSE curve
figure;
plot(1:epochs, MSEs, 'LineWidth',1.5);
xlabel('Iteration'); ylabel('Total Training MSE');
title('\alpha = 0.01, dropped features 1,2,3');
grid on;

% Compute confusion matrices and error rates
% Training
confTrain = zeros(C);
for k = 1:totSamples
    zk = W*trainSet(:,k) + b;
    [~, pred] = max(sigmoid(zk));
    trueC = ceil(k/NumTrainC);
    confTrain(trueC, pred) = confTrain(trueC, pred) + 1;
end
errTrain = 1 - trace(confTrain)/(totSamples);

disp('Confusion matrix (train):'); disp(confTrain);
disp(['Error rate (train): ', num2str(errTrain)]);

% Testing
totTest = size(testSet,2);
confTest = zeros(C);
for k = 1:totTest
    zk = W*testSet(:,k) + b;
    [~, pred] = max(sigmoid(zk));
    trueC = ceil(k/NumTestC);
    confTest(trueC, pred) = confTest(trueC, pred) + 1;
end
errTest = 1 - trace(confTest)/(totTest);

disp('Confusion matrix (test):'); disp(confTest);
disp(['Error rate (test): ', num2str(errTest)]);


% Create confusion chart for training set
figure;
cm_train = confusionchart(confTrain, ...
    'RowSummary','row-normalized', ...
    'ColumnSummary','column-normalized');
cm_train.Title = 'Confusion Matrix - Training Set';
cm_train.FontSize = 14;
cm_train.XLabel = 'Predicted Class';
cm_train.YLabel = 'True Class';

% Create confusion chart for test set
figure;
cm_test = confusionchart(confTest, ...
    'RowSummary','row-normalized', ...
    'ColumnSummary','column-normalized');
cm_test.Title = 'Confusion Matrix - Test Set';
cm_test.FontSize = 14;
cm_test.XLabel = 'Predicted Class';
cm_test.YLabel = 'True Class';



