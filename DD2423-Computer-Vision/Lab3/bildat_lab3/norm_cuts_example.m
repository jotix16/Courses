colour_bandwidth = 20.0; % color bandwidth
radius = 3;              % maximum neighbourhood distance
ncuts_thresh = 0.2;      % cutting threshold
min_area = 200;          % minimum area of segment
max_depth = 8;           % maximum splitting depth
scale_factor = 0.4;      % image downscale factor
image_sigma = 2.0;       % image preblurring scale


%% orange
radius = 6;
ncuts_thresh = 0.6;        % cutting threshold
min_area = 40;          % minimum area of segment
max_depth = 6;           % depth of rekursion

I = imread('orange.jpg');
I = imresize(I, scale_factor);
Iback = I;
d = 2*ceil(image_sigma*2) + 1;
h = fspecial('gaussian', [d d], image_sigma);
I = imfilter(I, h);
segm = norm_cuts_segm(I, colour_bandwidth, radius, ncuts_thresh, min_area, max_depth);
Inew = mean_segments(Iback, segm);
I = overlay_bounds(Iback, segm);
figure(1)
subplot(1,2,1)
imshow(Inew)
title('Segments');
subplot(1,2,2)
imshow(I)
title('Clusters');
sgtitle(['radius=',num2str(radius),' ncuts-thresh=',num2str(ncuts_thresh), ' min-area=',num2str(min_area), ' max-depth=', num2str(max_depth)])

%% tiger 1
radius = 7;
ncuts_thresh = 0.3;      % cutting threshold
min_area = 8;          % minimum area of segment
max_depth = 7;           % depth of rekursion


I = imread('tiger1.jpg');
I = imresize(I, scale_factor);
Iback = I;
d = 2*ceil(image_sigma*2) + 1;
h = fspecial('gaussian', [d d], image_sigma);
I = imfilter(I, h);
segm = norm_cuts_segm(I, colour_bandwidth, radius, ncuts_thresh, min_area, max_depth);
Inew = mean_segments(Iback, segm);
I = overlay_bounds(Iback, segm);

figure(2)
subplot(1,2,1)
imshow(Inew)
title('Segments');
subplot(1,2,2)
imshow(I)
title('Clusters');
sgtitle(['radius=',num2str(radius),' ncuts-thresh=',num2str(ncuts_thresh), ' min-area=',num2str(min_area), ' max-depth=', num2str(max_depth)])


%% tiger 2
radius = 10;
ncuts_thresh = 0.4;      % cutting threshold
min_area = 60;          % minimum area of segment
max_depth = 8;           % depth of rekursion


I = imread('tiger2.jpg');
I = imresize(I, scale_factor);
Iback = I;
d = 2*ceil(image_sigma*2) + 1;
h = fspecial('gaussian', [d d], image_sigma);
I = imfilter(I, h);
segm = norm_cuts_segm(I, colour_bandwidth, radius, ncuts_thresh, min_area, max_depth);
Inew = mean_segments(Iback, segm);
I = overlay_bounds(Iback, segm);

figure(3)
subplot(1,2,1)
imshow(Inew)
title('Segments');
subplot(1,2,2)
imshow(I)
title('Clusters');
sgtitle(['radius=',num2str(radius),' ncuts-thresh=',num2str(ncuts_thresh), ' min-area=',num2str(min_area), ' max-depth=', num2str(max_depth)])


%% tiger 3
radius = 5;
ncuts_thresh = 0.6;      % cutting threshold
min_area = 20;          % minimum area of segment
max_depth = 7;           % depth of rekursion


I = imread('tiger3.jpg');
I = imresize(I, scale_factor);
Iback = I;
d = 2*ceil(image_sigma*2) + 1;
h = fspecial('gaussian', [d d], image_sigma);
I = imfilter(I, h);
segm = norm_cuts_segm(I, colour_bandwidth, radius, ncuts_thresh, min_area, max_depth);
Inew = mean_segments(Iback, segm);
I = overlay_bounds(Iback, segm);

figure(4)
subplot(1,2,1)
imshow(Inew)
title('Segments');
subplot(1,2,2)
imshow(I)
title('Clusters');
sgtitle(['radius=',num2str(radius),' ncuts-thresh=',num2str(ncuts_thresh), ' min-area=',num2str(min_area), ' max-depth=', num2str(max_depth)])

