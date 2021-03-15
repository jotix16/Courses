function pic = gaussfft(pic,t)
%  Filtered with gaussian filter with variance t

% 1) Discretized gaussian filter
[m,n] = size(pic);
[X, Y] = meshgrid(-m / 2 : (m / 2) - 1, -n / 2 : (n / 2) - 1);
gauss = (1 / (2 * pi * t)) * exp(-(X .* X + Y .* Y) / (2 * t));
% 2) Convolution in FFT

pichat = fft2(fftshift(gauss)) .* fft2(pic);
% 3) Inverse FFT
pic = real(ifft2(pichat));

end

