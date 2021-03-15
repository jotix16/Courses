close all
%% Question 4
tools = few256;
scale = [0.0001, 1.0, 4.0, 16.0  64.0];
X = size(scale,2);

figure
subplot(2,3,1)
showgrey(tools)
title('Original image')
for i = 1:X
    subplot(2,3,i+1)
    contour(Lvvtilde(discgaussfft(tools, scale(i) ), 'same'), [0 0])
    title(['Scale = ',num2str(scale(i))])
    axis('image')
    axis('ij')
end

%% Question 5
tools = few256;
scale = [0.0001, 1.0, 4.0, 16.0  64.0];
X = size(scale,2);

figure
subplot(2,3,1)
showgrey(tools)
title('Original image')
for i = 1:X
    subplot(2,3,i+1)
    showgrey(Lvvvtilde(discgaussfft(tools, scale(i)), 'same') < 0)
    title(['Scale = ',num2str(scale(i))])
    axis('image')
    axis('ij')
end

%% Question 7
%%%%%%%%%%%% House %%%%%%%%%%%%
close all
house = godthem256;
scale = [7 12 25];
thresholds = [300 400 500 1000];
X = size(scale,2);
Y = size(thresholds,2);


figure
annotation('textbox', [0, 0.86, 0, 0], 'string', ['Threshold=',num2str(thresholds(1))])
annotation('textbox', [0, 0.64, 0, 0], 'string', ['Threshold=',num2str(thresholds(2))])
annotation('textbox', [0, 0.42, 0, 0], 'string', ['Threshold=',num2str(thresholds(3))])
annotation('textbox', [0, 0.21, 0, 0], 'string', ['Threshold=',num2str(thresholds(4))])

for i = 1:Y
    for j = 1:X
        subplot(Y,X,j+(i-1)*X)
        overlaycurves(house, extractedge(house, scale(j), thresholds(i), 'same'))
        title(['Scale = ',num2str(scale(j))])
    end
end

%%%%%%%%%%%% Tools %%%%%%%%%%%%
tools = few256;
scale = [7 12 25];
thresholds = [500 600 800 1000];
X = size(scale,2);
Y = size(thresholds,2);


figure
annotation('textbox', [0, 0.86, 0, 0], 'string', ['Threshold=',num2str(thresholds(1))])
annotation('textbox', [0, 0.64, 0, 0], 'string', ['Threshold=',num2str(thresholds(2))])
annotation('textbox', [0, 0.42, 0, 0], 'string', ['Threshold=',num2str(thresholds(3))])
annotation('textbox', [0, 0.21, 0, 0], 'string', ['Threshold=',num2str(thresholds(4))])

for i = 1:Y
    for j = 1:X
        subplot(Y,X,j+(i-1)*X)
        overlaycurves(tools, extractedge(tools, scale(j), thresholds(i), 'same'))
        title(['Scale = ',num2str(scale(j))])
    end
end

%%
close all
subplot(1,2,1)
overlaycurves(house, extractedge(house, 7,700 , 'same'))
title("Scale=7, Threshold=1000")
subplot(1,2,2)
overlaycurves(tools, extractedge(tools, 12,1000 , 'same'))
title("Scale=25, Threshold=1000")
