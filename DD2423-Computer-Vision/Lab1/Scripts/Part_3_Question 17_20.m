close all
%% 3.1 Smoothing of noisy data
office = office256;
add = gaussnoise(office, 16);
sap = sapnoise(office, 0.1, 255);
images = {add sap};

subplot(1,2,1)
showgrey(add)
title("Gaussian noise")
subplot(1,2,2)
showgrey(sap)
title("Salt & Pepper noise")

%% Gaussian filter
figure
annotation('textbox', [0, 0.86, 0, 0], 'string', ['Gaussian Noise'])
annotation('textbox', [0, 0.35, 0, 0], 'string', ['Salt & Pepper Noise'])
T = [1 4 16 64 256];  %  Different variance parameters
i = 1;
for j = 1:2
    for t= T
        imghat = gaussfft(images{j}, t);
        subplot(2,size(T,2),i)
        showgrey(imghat);
        title("t="+t)
        i = i+1;
    end
end

%% Median Filter
figure
annotation('textbox', [0, 0.86, 0, 0], 'string', ['Gaussian Noise'])
annotation('textbox', [0, 0.35, 0, 0], 'string', ['Salt & Pepper Noise'])
T = [2 4 8 10 20];  %  Different window-size parameters
i = 1;
for j = 1:2
    for t= T
        imghat = medfilt(images{j}, t);
        subplot(2,size(T,2),i)
        showgrey(imghat);
        title("t="+t)
        i = i+1;
    end
end

%% Ideal Low-Pass Filter
figure
annotation('textbox', [0, 0.86, 0, 0], 'string', ['Gaussian Noise'])
annotation('textbox', [0, 0.35, 0, 0], 'string', ['Salt & Pepper Noise'])
T = [0.05 0.1 0.2 0.3 1];  %  Different cut-off frequency parameters
i = 1;
for j = 1:2
    for t= T
        imghat = ideal(images{j}, t);
        subplot(2,size(T,2),i)
        showgrey(imghat);
        title("t="+t)
        i = i+1;
    end
end


%% 3.2 Smoothing and subsampling
img = phonecalc256;
imgtemp = img;
smoothing = img;
smoothing2 = img;
N=5;
for i=1:N
    if i>1
        % generate subsampled versions
        img = rawsubsample(img);
        smoothing = gaussfft(smoothing,0.6);    
        smoothing = rawsubsample(smoothing);
        smoothing2 = ideal(smoothing2,0.3);
        smoothing2 = rawsubsample(smoothing2);
    end
    subplot(3, N, i)
    showgrey(img)
    title("Sampledx"+i)
    subplot(3, N, i+N)
    showgrey(smoothing)
    title("Smoothed(t=0.6)x"+i+" & sampledx"+i)
    subplot(3, N, i+2*N)
    showgrey(smoothing2)
    title("Smoothedx(t=0.3)"+i+" & sampledx"+i)
end








