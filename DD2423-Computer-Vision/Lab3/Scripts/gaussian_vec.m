function g = gaussian_vec(pixels,mean,cov)
%UNTITLED4 Summary of this function goes here
%   Detailed explanation goes here
diff = pixels - mean;
g = 1 / sqrt(det(cov) * (2 * pi)^3) * exp(-0.5 * sum((diff * inv(cov) .* diff), 2));
end

