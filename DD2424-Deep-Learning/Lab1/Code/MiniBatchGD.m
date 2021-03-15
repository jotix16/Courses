function [Wstar, bstar] = MiniBatchGD(X, Y, GDparams, W, b, lambda,svm)
%% Does one Gradient Descent update based on a minibatch.
% Input:
%  - X:             [DxN] D is dimensionality of each sample and N is the
%                   nr of samples in the minibatch.
%   
%  - Y:             [KxN] one hot representation of the labels of each image. 
%                   K is nr of possible classes.
%
%  - GDparams:      Holds the hyperparameters:
%                       GDparams.n_batch- size of each minibatch 
%                       GDparams.n_epoch- number of epochs
%                       GDparams.eta- learningrate
%
%  - W:             [KxD] weights matrix
%
%  - b:             [Kx1] bias
%
%  - lambda:        [double] regularizazion param
%
%  - svm:           [bool] is true if we are using hinge loss
%
% Output:
%  - W_star:        [Nx1] vector of labels for each image
%
%  - b_star:        [Nx1] vector of labels for each image
%
%%
    if nargin<7
      svm = false;
    end

    n = size(X,2);
    Wstar = W;
    bstar = b;
    n_batch = GDparams.n_batch;
    
    for j=1:n/n_batch
        j_start = (j-1)*n_batch + 1;
        j_end = j*n_batch;
        Xbatch = X(:, j_start:j_end);
        Ybatch = Y(:, j_start:j_end);
        
        % Compute gradients for the batch
        P = EvaluateClassifier(Xbatch, Wstar, bstar);
        if svm
            [grad_W, grad_b] = ComputeGradientsSVM(Xbatch, Ybatch, Wstar,bstar, lambda); 
        else
            [grad_W, grad_b] = ComputeGradients(Xbatch, Ybatch, P, Wstar, lambda);      
        end
        
        % Update Weights
        Wstar = Wstar - GDparams.eta*grad_W;
        bstar = bstar - GDparams.eta*grad_b;
    end
end