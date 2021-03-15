%% %%%% TOOLS %%%%% %%

close all;
tools = few256;
toolsgauss = discgaussfft(tools,0.6);

% Not changed tools
subplot(2,2,1)
showgrey(tools)
title("Image")

subplot(2,2,2)
showgrey(toolsgauss)
title("Smoothed image")


% Thresholded tools
threshold = 10500;
gradmagntools = Lv(tools);
gradmagntoolsgauss = Lv(toolsgauss);
% maks= max(gradmagntools(:));
% threshold = maks * 0.017;
% maksgauss= max(gradmagntoolsgauss(:));

subplot(2,2,3)
showgrey((gradmagntools - threshold) > 0)
title("Magnitude of gradient")

subplot(2,2,4)
showgrey((gradmagntoolsgauss - threshold) > 0)
title("Magnitude of gradient after smoothing")


%% %%%% HOUSE %%%% %%
close all;
tools = godthem256;
toolsgauss = discgaussfft(tools,0.6);


% Not changed house
figure
subplot(2,2,1)
showgrey(tools)
title("Image")

subplot(2,2,2)
showgrey(toolsgauss)
title("Smoothed image")


% Thresholding house
threshold = 18500;
gradmagntools = Lv(tools);
gradmagntoolsgauss = Lv(toolsgauss);

subplot(2,2,3)
showgrey((gradmagntools - threshold) > 0)
title("Magnitude of gradient")

subplot(2,2,4)
showgrey((gradmagntoolsgauss - threshold) > 0)
title("Magnitude of gradient after smoothing")
