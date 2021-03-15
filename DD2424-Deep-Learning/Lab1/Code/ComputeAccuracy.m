function acc = ComputeAccuracy(X, y, W, b,svm)
%% Computes Acciracy for a given test set.
% Input:
%  - X:             [DxN] D is dimensionality of each sample and N is the
%                   nr of samples in the minibatch.
%   
%  - y:             [1xN] labels for each of N samples.
%                   or
%                   [KxN] one hot representation of the labels.
%
%  - W:             [KxD] weights matrix
%
%  - b:             [Kx1] bias
%
%  - svm:           [bool] is true if we are using hinge loss, 
%                   rather than Cross-Entropy.
%
% Output:
%  - acc:           [double] accuracy 0<acc<1 
%
%%
if nargin<5
  svm = false;
end

if svm
    % SVM Classification
    [~,classes] = max(W*X+b);
else
    % Softmax Classification
    P = EvaluateClassifier(X, W, b);
    [~,classes] = max(P);
end

if size(y,1)>1
    y=vec2ind(y);
end
acc = sum(classes==y)/length(y);

end

