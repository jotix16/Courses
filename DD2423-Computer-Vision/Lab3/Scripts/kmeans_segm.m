function [ segmentation, centers ] = kmeans_segm(image, K, L, seed)
%UNTITLED Summary of this function goes here
%   K                   nr of clusters
%   L                   nr of iterations
%   seed                seed for the random generator

%% Used variables:
% V                 [K x 3] matrix having the centers of the clusters as columns
% image2            [X*Y x 4] where the last column gives the cluster if each pixel
rng(seed)
dim = ndims(image);
if dim ==3
    [X,Y,Z]=size(image);
    image2 = double([reshape(image, X*Y,Z), zeros(X*Y,1)]);
else
    [X,Y] =size(image);
    image2 = double([image, zeros(X,1)]);
end

% Randomly initialize the K cluster centers
% centers = randi([0 255],K,3,'like',image);
centers = zeros(K, 3);
idx = randperm(size(image2, 1), K);
for i = 1 : K
    centers(i, :) = image2(idx(i), 1:3);
end

% Iterate L times
for l=1:L
    % Compute all distances between pixels and cluster centers
    [~,I] = pdist2(double(centers),image2(:,1:3),'euclidean' ,'Smallest',1);
    % Check the changes hear [TODO]
    % Assign each pixel to the cluster center for which the distance is minimum
    image2(:,4) = I';
    % Recompute each cluster center by taking the mean of all pixels assigned to it
    for k =1:K
        indeces = image2(:,4)==k;
        if sum(indeces)
            centers(k,:) = sum(image2(indeces,1:3))/sum(indeces);
        end
    end
end

if dim == 3
    segmentation = reshape(image2(:,4),X,Y);
else
    segmentation = image2(:,4);
end
end
