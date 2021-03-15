% This function calcultes the weights for each particle based on the
% observation likelihood
%           S_bar(t)            4XM
%           outlier             1Xn
%           Psi(t)              1XnXM
% Outputs: 
%           S_bar(t)            4XM
function S_bar = weight(S_bar, Psi, outlier)

    % YOUR IMPLEMENTATION
    Psi=Psi(1,find(~outlier),:);
    x = prod(Psi,2);
%     x(1,1,outlier) = 0; % no weight for outliers
    x = x/(sum(x)); % normalize
    S_bar(4,:) = x;
end
