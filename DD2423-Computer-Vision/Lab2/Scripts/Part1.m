tools = few256;
showgrey(few256)
title('Given Image')

%% Create figure 
figure
annotation('textbox', [0, 0.86, 0, 0], 'string', 'Central diffrences')
annotation('textbox', [0, 0.6, 0, 0], 'string', 'simple diffrence operator')
annotation('textbox', [0, 0.3, 0, 0], 'string', 'Sobel operator')

% a) Central differences
deltax = [1 0 -1];
deltay = [1; 0; -1];

dxtools = conv2(tools, deltax, 'valid');
dytools = conv2(tools, deltay, 'valid');
subplot(3,2,1)
showgrey(dxtools)
title("In x-direction")
ylabel('Central diffrences')
subplot(3,2,2)
showgrey(dytools)
title("In y-direction")
% b) Simple difference operator
deltax = [1 -1];
deltay = [1; -1];

dxtools = conv2(tools, deltax, 'valid');
dytools = conv2(tools, deltay, 'valid');
subplot(3,2,3)
showgrey(dxtools)
ylabel('simple diffrence operator')
subplot(3,2,4)
showgrey(dytools)

% c) Sobel operator(lowpass filterzing perpendicular to directional differentation in addition)
deltax = [1 0 -1; 2 0 -2; 1 0 -1];
deltay = deltax';

dxtools = conv2(tools, deltax, 'valid');
dytools = conv2(tools, deltay, 'valid');
subplot(3,2,5)
showgrey(dxtools)
ylabel('Sobel operator')
subplot(3,2,6)
showgrey(dytools)