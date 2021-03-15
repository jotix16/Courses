scale_factor = 0.5;       % image downscale factor
spatial_bandwidth = 10.0;  % spatial bandwidth
colour_bandwidth = 5.0;   % colour bandwidth
num_iterations = 40;      % number of mean-shift iterations
image_sigma = 1.0;        % image preblurring scale

I = imread('tiger1.jpg');
I = imresize(I, scale_factor);
Iback = I;
d = 2*ceil(image_sigma*2) + 1;
h = fspecial('gaussian', [d d], image_sigma);
I = imfilter(I, h);

%%
spatial_bandwidth = [ 6  20];
% colour_bandwidth = [3 4 5 6 7 8];
colour_bandwidth = 5;
for i =1:2
    segm = mean_shift_segm(I, spatial_bandwidth(i), colour_bandwidth, num_iterations);
    Inew = mean_segments(Iback, segm);
    Iseg = overlay_bounds(Iback, segm);
    figure(1)
    subplot(2,2,i); imshow(Iseg); title(['\sigma_s=',num2str(spatial_bandwidth(i)), ', \sigma_c=',num2str(colour_bandwidth)])
    subplot(2,2,i+2); imshow(Inew); title(['\sigma_s=',num2str(spatial_bandwidth(i)), ', \sigma_c=',num2str(colour_bandwidth)])

end


%%
spatial_bandwidth = 4;
colour_bandwidth = [0.5 30];
for i =1:2
    segm = mean_shift_segm(I, spatial_bandwidth, colour_bandwidth(i), num_iterations);
    Inew = mean_segments(Iback, segm);
    Iseg = overlay_bounds(Iback, segm);
    figure(3)
    subplot(2,2,i+2); imshow(Iseg); title(['\sigma_s=',num2str(spatial_bandwidth),', \sigma_c=',num2str(colour_bandwidth(i))])
    subplot(2,2,i); imshow(Inew); title(['\sigma_s=',num2str(spatial_bandwidth),', \sigma_c=',num2str(colour_bandwidth(i))])

end

