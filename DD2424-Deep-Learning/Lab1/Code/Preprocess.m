function X = Preprocess(X)
%% Preprocesses training samples so they have 0 mean and 1 std.
% Input:
%  - X:     [DxN] where D is the feature dimension and N is the number
%           of samples.
% Output:
%  - X:     [DxN] preprocessed training samples.
%%
    mean_X = mean(X, 2);
    std_X = std(X, 0, 2);
    
    X = X - repmat(mean_X, [1, size(X, 2)]);
    X = X ./ repmat(std_X, [1, size(X, 2)]);
end