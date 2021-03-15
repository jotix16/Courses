function [X, Y, y] = LoadBatch(filename)
%% Loads training data from file
% Input:
%  - filename: string that contains the name of the file to load from
%
% Output:
%  - X:      [DxN] containst image pixel data rowise. D is the dimensionality
%           of each image and N is the number of images
%   
%  - Y:      [KxN] one hot representation of the label of each image. 
%            K is nr of possible classes(labels)
%
%  - y:      [Nx1] vector of labels for each image
%   
%%
    A = load(filename);
    N = size(A.data,1);
    X = double(A.data');
    Y=full(ind2vec(double(A.labels'+1)));
    y = double(A.labels'+1);
end