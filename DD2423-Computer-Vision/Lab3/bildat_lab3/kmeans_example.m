K = 3;               % number of clusters used
L = 55;              % number of iterations
seed = 14;           % seed used for random initialization
scale_factor = 1;  % image downscale factor
image_sigma = 0.002;   % image preblurring scale

I = imread('orange.jpg');
I = imresize(I, scale_factor);
Iback = I;
d = 2*ceil(image_sigma*2) + 1;
h = fspecial('gaussian', [d d], image_sigma);
I = imfilter(I, h);

%% Different cluster number K
close all
K = [3,5];
tic
for i = 1:size(K,2)
    
    [ segm, centers ] = kmeans_segm(I, K(i), L, seed);
    Inew = mean_segments(Iback, segm);
    Iseg = overlay_bounds(Iback, segm);
    subplot(2,2,2*(i-1)+1)
    imshow(Iseg)
    ylabel(['K = ',num2str(K(i))] )
    set(get(gca,'YLabel'),'Rotation',0)
    if i==1; title('Segments'); end
    subplot(2,2,2*(i-1)+2)
    imshow(Inew)
    if i==1; title('Clusters'); end
end
toc

figure
K = [6,8];
tic
for i = 1:size(K,2)
    
    [ segm, centers ] = kmeans_segm(I, K(i), L, seed);
    Inew = mean_segments(Iback, segm);
    Iseg = overlay_bounds(Iback, segm);
    subplot(2,2,2*(i-1)+1)
    imshow(Iseg)
    ylabel(['K = ',num2str(K(i))] )
    set(get(gca,'YLabel'),'Rotation',0)
    if i==1; title('Segments'); end
    subplot(2,2,2*(i-1)+2)
    imshow(Inew)
    if i==1; title('Clusters'); end
end
toc

%% Different iteration number L
K = 3;
L = [3,4];
tic
for i = 1:size(L,2)
    
    [ segm, centers ] = kmeans_segm(I, K, L(i), seed);
    Inew = mean_segments(Iback, segm);
    Iseg = overlay_bounds(Iback, segm);
    subplot(2,2,2*(i-1)+1)
    imshow(Iseg)
    ylabel(['L = ',num2str(L(i))] )
    set(get(gca,'YLabel'),'Rotation',0)
    if i==1; title('Segments'); end
    subplot(2,2,2*(i-1)+2)
    imshow(Inew)
    if i==1; title('Clusters'); end
end
toc

figure
L = [5,8];
tic
for i = 1:size(L,2)
    
    [ segm, centers ] = kmeans_segm(I, K, L(i), seed);
    Inew = mean_segments(Iback, segm);
    Iseg = overlay_bounds(Iback, segm);
    subplot(2,2,2*(i-1)+1)
    imshow(Iseg)
    ylabel(['L = ',num2str(L(i))] )
    set(get(gca,'YLabel'),'Rotation',0)
    if i==1; title('Segments'); end
    subplot(2,2,2*(i-1)+2)
    imshow(Inew)
    if i==1; title('Clusters'); end
end
toc

%% Tiger
K = 8;               % number of clusters used
L = 25;              % number of iterations
seed = 14;           % seed used for random initialization
scale_factor = 1;  % image downscale factor
image_sigma = 1.4;   % image preblurring scale

I = imread('tiger1.jpg');
I = imresize(I, scale_factor);
Iback = I;
d = 2*ceil(image_sigma*2) + 1;
h = fspecial('gaussian', [d d], image_sigma);
I = imfilter(I, h);

[ segm, centers ] = kmeans_segm(I, K, L, seed);
Inew = mean_segments(Iback, segm);
Iseg = overlay_bounds(Iback, segm);
subplot(1,2,1)
imshow(Iseg)
ylabel(['K=',num2str(K), ',L=',num2str(L) ])
set(get(gca,'YLabel'),'Rotation',0)
title('Segments')
subplot(1,2,2)
imshow(Inew)
title('Clusters')