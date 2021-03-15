function J = ComputeCost(X, Y, W, b, lambda)
%% Computes Cross-Entropy cost function for a given minibatch.
% Input:
%  - X:         [DxN] D is the dimensionality of each image sample 
%               and N is the number of samples on the minibatch
%   
%  - Y:         [KxN] one hot representation of the label of each sample. 
%               K is nr of possible classes.
%
%  - W:         [KxD] weights
%
%  - b:         [Kx1] bias
%
%  - lambda:    [double] regularization parameter
%
% Output:
%  - J:         [double] cost
%%
P = EvaluateClassifier(X, W, b);
l = -log(sum(P.*Y,1));
J = sum(l)/size(X,2) + 0.5*lambda*sum(sum(abs(W).^2));
end