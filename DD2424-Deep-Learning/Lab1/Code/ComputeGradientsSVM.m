function [grad_W, grad_b] = ComputeGradientsSVM(X, Y, W, b, lambda)
%% Computes gradients for a given minibatch based on SVM(hinge) loss function
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
%  - grad_W:    [KxD] gradients for the weights
%
%  - grad_b:    [KxD] gradient for the bias
%%
% initialize gradients
N = size(X,2);
[K,D] = size(W);
grad_W = zeros(K,D);
grad_b = zeros(K,1);

% Add gradients for each sample
for i=1:N
    % Calc grad for the output of the neurons before softmax for sample i
    [~,y] = max(Y(:,i));
    s = W*X(:,i)+b;
    grad_s = (s -s(y)-1)>0;
    
    % Calc gradients for bias and weights
    grad_b = grad_b + grad_s;
    grad_W = grad_W + grad_s * X(:,i)';
end
% Normalize and add  gradient of regularization
grad_b = grad_b./size(X,2);
grad_W = grad_W./size(X,2) + 2*lambda*W;
end

