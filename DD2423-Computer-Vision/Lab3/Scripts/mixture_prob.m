function prob = mixture_prob(image, K, L, mask)
%UNTITLED Summary of this function goes here

%% input
% image
% K
% L
% mask
%% help variables
% covs,means,weights represent the K gaussian mixtures
% covs          [3x3xK]
% means         [Kx3]
% weights       [Kx1]
% P             [m*n K]    holds probabilities for each pixel to come from
%                                                            K-th mixture.
% pixels        [m*n p]

%% output  
% prob          [m*n 1]    hold the probability of color of pixel i
% prob = weigts' * [gaus1(pixels) ... gausk(pixels)] where pixels are [m*n 1]
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% solution
% Store all pixels for which mask=1 in a Nx3 matrix
[m,n,p] = size(image);
pixels = im2double(reshape(image,m*n,p));
mask = logical(reshape(repmat(mask,1,1,3),m*n,p));
pixels = pixels(mask);
pixels = reshape(pixels,length(pixels)/3,3);



%% Let I be a set of pixels and V be a set of K Gaussian components in 3D (R,G,B).
% Randomly initialize the K components using masked pixels
seed = 5;
covs = double(repmat(eye(3),1,1,K).*rand);
weights = zeros(K,1);

[ seg, means ] = kmeans_segm(pixels, K, L, seed); % run K-mean with K clusters

for k =1:K
    weights(k) = sum(seg==k)/size(seg,1);
end

%%
N = size(pixels,1);
prob = zeros(N,K);
%% Iterate L times
for i = 1:L
%% Expectation: Compute probabilities P_ik using masked pixels
    for k = 1:K
            prob(:,k) = gaussian_vec(pixels,means(k,:),covs(:,:,k));
    end
    prob = prob .* weights';
    normalizer = sum(prob,2);
    prob = prob ./ normalizer;

%% Maximization: Update weights, means and covariances using masked pixels   
    total = sum(prob,1)';
    weights = total ./size(prob,1);
    means = (prob' * pixels) ./ sum(prob,1)';
    for k = 1:K
        temp1 = pixels - means(k,:);
        temp2 = prob(:,k).* temp1;
        covs(:,:,k) = (temp2' * temp1) / total(k);   % HERE CAN BE BE THE ERROR
    end
end

%% Compute probabilities p(c_i) in Eq.(3) for all pixels I.
[m,n,p] = size(image);
pixels = im2double(reshape(image,m*n,p));
p = zeros(m*n,K);
%%
for k = 1:K

      p(:,k) = weights(k) .* gaussian_vec(pixels,means(k,:),covs(:,:,k));
end
prob = reshape(sum(p,2),m,n,1);
end

