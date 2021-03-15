% Assignement1 (Basic)
    %% Load Data
    [X_train, Y_train, y_train] = LoadBatch('data_batch_1.mat');
    [X_valid, Y_valid, y_valid] = LoadBatch('data_batch_2.mat');
    [X_test, Y_test, y_test] = LoadBatch('test_batch.mat');
    d = size(X_train,1);
    K = size(Y_train,1);
    %% Load Full Data
%     [X_train, Y_train, X_valid, Y_valid,X_test, Y_test] = loadFullBatch();
%     d = size(X_train,1);
%     K = size(Y_train,1);
    
    %% Preprocess
    X_train = Preprocess(X_train);
    X_test = Preprocess(X_test);
    X_valid = Preprocess(X_valid);
    
    %% Initialize Params
    %rng(400)
    std = 0.01;
    W = randn(K,d)*std;
    b = randn(K,1)*std;
    
    %% Right gradients
%     [ngrad_b, ngrad_W] = ComputeGradsNumSlow(X(1:20, 1), Y(:, 1),W(:,1:20), b, lambda, 1e-6);
%     [ngrad_W2,ngrad_b2] = ComputeGradients(X(1:20, 1), Y(:, 1),EvaluateClassifier(X(1:20, 1), W(:,1:20), b), W(:,1:20), lambda);
%     
%     norm(ngrad_b2-ngrad_b)<1e-3
%     norm(ngrad_W2-ngrad_W)<1e-3

    %% Train
    lambda = 0;
    GDparams.n_epoch = 40;
    GDparams.n_batch= 100;
    GDparams.eta = .001;
    
    
    train_cost = zeros(1,GDparams.n_epoch);
    valid_cost = zeros(1,GDparams.n_epoch);
    for i = 1:GDparams.n_epoch
        fprintf("Epoche: " +num2str(i)+"\n")
        % Shuffling
        mix = randperm(size(X_train,2));
        X_train = X_train(:,mix);
        Y_train = Y_train(:,mix);
        % Minibatch Training
        [W, b] = MiniBatchGD(X_train, Y_train, GDparams, W, b, lambda);
        
        % Debug info
        train_cost(i) = ComputeCost(X_train, Y_train, W, b, lambda);
        valid_cost(i) = ComputeCost(X_valid, Y_valid, W, b, lambda);
    end
    
    %% Quality
    plot(1:GDparams.n_epoch, train_cost, 1:GDparams.n_epoch, valid_cost)
    title("The cost function"+": lambda="+num2str(lambda)+", nr epochs="+num2str(GDparams.n_epoch)+", nr batch="+num2str(GDparams.n_batch)+", eta="+num2str(GDparams.eta))
    xlabel('epoch nr')
    ylabel('Cost')
    legend("Training Cost","Validation Cost")
    acc_train = ComputeAccuracy(X_train, Y_train, W, b);
    acc_valid = ComputeAccuracy(X_valid, Y_valid, W, b);
    acc_test = ComputeAccuracy(X_test, Y_test, W, b);
    fprintf("The cost function for"+" (lambda="+num2str(lambda)+", nr epochs="+num2str(GDparams.n_epoch)+", nr batch="+num2str(GDparams.n_batch)+", eta="+num2str(GDparams.eta)+") is:\n")
    disp(["acc_train" "acc_valid" "acc_test";acc_train acc_valid acc_test])
    
    %% Visualize 
    for i=1:10
        im = reshape(W(i, :), 32, 32, 3);
        s_im{i} = (im - min(im(:))) / (max(im(:)));
        s_im{i} = permute(s_im{i}, [2, 1, 3]);
    end
    figure
    montage(s_im,'Size', [1,10])

%% My function
function [X, Y, y] = LoadBatch(filename)
    %% 
    %   X:      [DxN] containst image pixel data rowise. D is the dimensionality
    %           of each image and N is the number of images
    %   
    %   Y:      [KxN] one hot representation of the label of each image. 
    %            K is nr of possible classes(labels)
    %
    %   y:      [Nx1] vector of labels for each image
    %   
    %
    %
    A = load(filename);
    N = size(A.data,1);
    X = double(A.data');
    Y=full(ind2vec(double(A.labels'+1)));
    y = double(A.labels'+1);

end

function X = Preprocess(X)
    mean_X = mean(X, 2);
    std_X = std(X, 0, 2);
    
    X = X - repmat(mean_X, [1, size(X, 2)]);
    X = X ./ repmat(std_X, [1, size(X, 2)]);
end

function P = EvaluateClassifier(X, W, b)
    Y = exp(W*X+b);
    P = Y./repmat(sum(Y,1),size(Y,1),1);
end

function J = ComputeCost(X, Y, W, b, lambda)
             
P = EvaluateClassifier(X, W, b);
l = -log(sum(P.*Y,1));
J = sum(l)/size(X,2) + 0.5*lambda*sum(sum(abs(W).^2));
end

function acc = ComputeAccuracy(X, y, W, b)
P = EvaluateClassifier(X, W, b);
[~,classes] = max(P);
if size(y,1)>1
    y=vec2ind(y);
end
acc = sum(classes==y)/length(y);
end

function [grad_W, grad_b] = ComputeGradients(X, Y, P, W, lambda)
% initialize gradients
N = size(X,2);
[K,D] = size(W);
grad_W = zeros(K,D);
grad_b = zeros(K,1);

% gradient p for all samples in batch(gradients are columns of Grad_p)
Grad_P = -Y.* (1./sum(Y.*P,1));

% Add gradients for each sample
for i=1:N
    Pi= P(:,i);
    % Calc grad for the output of the neurons before softmax for sample i
    J = diag(Pi)-Pi*Pi';
    grad_s = J * Grad_P(:,i);
    
    % Calc gradients for bias and weights
    grad_b = grad_b + grad_s;
    grad_W = grad_W + grad_s * X(:,i)';
end
grad_b = grad_b./size(X,2);
grad_W = grad_W./size(X,2) + 2*lambda*W;
end

function [Wstar, bstar] = MiniBatchGD(X, Y, GDparams, W, b, lambda)
    n = size(X,2);
    Wstar = W;
    bstar = b;
    n_batch = GDparams.n_batch;
    for j=1:n/n_batch
        j_start = (j-1)*n_batch + 1;
        j_end = j*n_batch;
        Xbatch = X(:, j_start:j_end);
        Ybatch = Y(:, j_start:j_end);
        P = EvaluateClassifier(Xbatch, Wstar, bstar);
        [grad_W, grad_b] = ComputeGradients(Xbatch, Ybatch, P, Wstar, lambda);
%             [grad_b2,grad_W2] = ComputeGradsNum(Xbatch, Ybatch, Wstar,bstar, lambda, 1e-6);
%             norm(grad_b2-grad_b)<1e-3
%             norm(grad_W-grad_W2)<1e-3           

        Wstar = Wstar - GDparams.eta*grad_W;
        bstar = bstar - GDparams.eta*grad_b;
    end
end

function [X_train, Y_train, X_valid, Y_valid,X_test, Y_test] = loadFullBatch()
%LOADFULLBATCH Summary of this function goes here
%%  Detailed explanation goes here

    filenames = ["data_batch_1.mat","data_batch_2.mat","data_batch_3.mat","data_batch_4.mat","data_batch_5.mat"]
    
    X = [];
    Y = [];
    y = [];
    for i=1:5
        filename = filenames(i);
        A = load(filename);
        N = size(A.data,1);
        X  = [X,double(A.data')];
        Y = [Y, full(ind2vec(double(A.labels'+1)))];
    end
      
    X_train = X(:,1001:end);
    Y_train = Y(:,1001:end);
    
    X_valid = X(:,1:1000);
    Y_valid = Y(:,1:1000);
    
    A = load('test_batch.mat');
    N = size(A.data,1);
    X_test  = double(A.data');
    Y_test = full(ind2vec(double(A.labels'+1)));
end


%% Numerical Gradients
function [grad_b, grad_W] = ComputeGradsNum(X, Y, W, b, lambda, h)

no = size(W, 1);
d = size(X, 1);

grad_W = zeros(size(W));
grad_b = zeros(no, 1);

c = ComputeCost(X, Y, W, b, lambda);

for i=1:length(b)
    b_try = b;
    b_try(i) = b_try(i) + h;
    c2 = ComputeCost(X, Y, W, b_try, lambda);
    grad_b(i) = (c2-c) / h;
end

for i=1:numel(W)   
    
    W_try = W;
    W_try(i) = W_try(i) + h;
    c2 = ComputeCost(X, Y, W_try, b, lambda);
    
    grad_W(i) = (c2-c) / h;
end
end

function [grad_b, grad_W] = ComputeGradsNumSlow(X, Y, W, b, lambda, h)

no = size(W, 1);
d = size(X, 1);

grad_W = zeros(size(W));
grad_b = zeros(no, 1);

for i=1:length(b)
    b_try = b;
    b_try(i) = b_try(i) - h;
    c1 = ComputeCost(X, Y, W, b_try, lambda);
    b_try = b;
    b_try(i) = b_try(i) + h;
    c2 = ComputeCost(X, Y, W, b_try, lambda);
    grad_b(i) = (c2-c1) / (2*h);
end

for i=1:numel(W)
    
    W_try = W;
    W_try(i) = W_try(i) - h;
    c1 = ComputeCost(X, Y, W_try, b, lambda);
    
    W_try = W;
    W_try(i) = W_try(i) + h;
    c2 = ComputeCost(X, Y, W_try, b, lambda);
    
    grad_W(i) = (c2-c1) / (2*h);
end
end
%%
