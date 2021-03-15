F = [ zeros(56, 128); ones(16, 128); zeros(56, 128)];
G = F';


%% 1.4 Linearity
H = F + 2 * G;

Fhat = fft2(F);
Ghat = fft2(G);
Hhat = fft2(H);

showgrey(F)
%% not centered
showgrey(log(1 + abs(Fhat)));
%% not centered
showgrey(log(1 + abs(Ghat)));
%% not centered
showgrey(log(1 + abs(Hhat)));
%% not centered
showgrey(log(1 + abs(fftshift(Hhat))));


%% 1.5 Pointwise multiplication
subplot(1,3,1)
showgrey(F .* G);
title("F=A.*B")

% Fourie transform of pointwise multiplication
subplot(1,3,2)
showfs(fft2(F.*G));
title("fft(F)")
% Pointwise multiplication corresponds to Multiplication of spectras
subplot(1,3,3)
showfs(Fhat*Ghat/128^2) ;
title("1/128^2*fft(A)*fft(B)")


%% 1.6 Scaling
F = [zeros(60, 128); ones(8, 128); zeros(60, 128)] .* ...
[zeros(128, 48) ones(128, 32) zeros(128, 48)];

subplot(1,2,1)
showgray(F)
subplot(1,2,2)
showfs(fft2(F))


%% 1.7 Rotating
F = [zeros(60, 128); ones(8, 128); zeros(60, 128)] .* ...
[zeros(128, 48) ones(128, 32) zeros(128, 48)];
subplot(3,2,1)
showgrey(F)

alpha = 90;
G = rot(F, alpha );  % rotate F by alpha
subplot(3,2,2)
showgrey(G)

Fhat = fft2(F);
subplot(3,2,3)
showfs(Fhat)
% showgrey(log(1 + fftshift(abs(Fhat))))

Ghat = fft2(G);
subplot(3,2,4)
showfs(Ghat)
% showgrey(log(1 + fftshift(abs(Fhat))))

Hhat = rot(fftshift(Ghat), -alpha );
subplot(3,2,5)
showgrey(log(1 + abs(Hhat)))  % already shifted, that is why showfs(Hhat) gives different figure


%% 1.8 Phase and magnitude information 
img1 = phonecalc128;
img2 = few128;
img3 = nallo128;

subplot(3,3,1)
showgrey(img1)
subplot(3,3,2)
showgrey(pow2image(img1, 1e-10 ))
subplot(3,3,3)
showgrey(randphaseimage(img1))

subplot(3,3,4)
showgrey(img2)
subplot(3,3,5)
showgrey(pow2image(img2, 1e-10 ))
subplot(3,3,6)
showgrey(randphaseimage(img1))

subplot(3,3,7)
showgrey(img3)
subplot(3,3,8)
showgrey(pow2image(img3, 1e-10 ))
subplot(3,3,9)
showgrey(randphaseimage(img1))