function [X_train, Y_train, X_valid, Y_valid,X_test, Y_test] = loadFullBatch()
%% Loads training data from data_batch_1.mat-data_batch_5.mat and test set 
% from test_batch.mat. The validation set contains 1000 samples randomly
% choosen from the training set.
%
% Output:
%  - X_train:      [DxN] D is the dimensionality of each sample and N is 
%                   the number of images
%   
%  - Y_train:      [KxN] one hot representation of the label of each image. 
%                   K is nr of possible classes(labels)
%
%  - X_valid:      [Dx1000] 
%   
%  - Y_valid:      [Kx1000] 
%
%  - X_test:       [DxM] 
%   
%  - Y_test:       [KxM] 
%%


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