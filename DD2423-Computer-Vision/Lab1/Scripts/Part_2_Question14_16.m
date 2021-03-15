%% Load images
img1 = phonecalc128;
img2 = few128;
img3 = nallo128;


%% Spatial discretization and spatial convolution
close all
% 1) Discrtized gaussian filter
t = 0.5;
g = fspecial('gaussian',[3,3],1/t );
% 2) convolution
pic = conv2(img1,g);
subplot(1,2,1)
showgrey(img1)
subplot(1,2,2)
showgrey(pic)


%% Spatial discretization and convolution via FFT
close all
% 1) Discretized gaussian filter
t = 0.5;
pic = gaussfft(img1,t);
subplot(1,2,1)
showgrey(img1)
subplot(1,2,2)
showgrey(pic)


%% Question 14 -- Add images to doc
T = [0.1 0.3 1 10 100];
i = 1;
for t=T
    tmp = gaussfft(deltafcn(128, 128), t);
    subplot(size(T,2)-2,2,i)
    showgrey(tmp);
    title("t="+t)
    variance(tmp)
    i = i+1;
end

%% Question 16 -- Add images to doc
T = [1 4 16 64 256];
images = {phonecalc128 few128 nallo128};
i = 1;
for t= T
    for j = 1:3
        imghat = gaussfft(images{j}, t);
        subplot(size(T,2),3,i)
        showgrey(imghat);
        title("t="+t)
        i = i+1;
    end
end