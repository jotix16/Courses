function P = EvaluateClassifier(X, W, b)
%% Evaluates X with params W and b.
% Input:
%  - X:     [DxN] where D is the feature dimension and N is the number
%           of samples to be evaluated.
%
%  - Y:     [KxN] one hot representation of the label of each sample. 
%           K is nr of possible classes.
%
%  - W:     [KxD] weights
%
% Output:
%  - P:     [KxN] softmax evaluation if NN
%%
    Y = exp(W*X+b);
    P = Y./repmat(sum(Y,1),size(Y,1),1);
end