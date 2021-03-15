% This function performs the maximum likelihood association and outlier detection.
% Note that the bearing error lies in the interval [-pi,pi)
%           mu_bar(t)           3X1
%           sigma_bar(t)        3X3
%           z(t)                2Xn
% Outputs: 
%           c(t)                1Xn
%           outlier             1Xn
%           nu_bar(t)           2nX1
%           H_bar(t)            2nX3
function [c, outlier, nu_bar, H_bar] = batch_associate(mu_bar, sigma_bar, z)
        
        % YOUR IMPLEMENTATION %
    
    [m,n] = size(z);
    c = zeros(1,n);
    outlier = zeros(1,n);
    nu_bar = [];
    H_bar = [];
    
    for i= 1:n
       [c_tmp, outlier_tmp, nu, S, H] = associate(mu_bar, sigma_bar, z(:,i));
       c(1,i) = c_tmp;
       outlier(1,i) = outlier_tmp;
       nu_bar = [nu_bar; nu(:,c_tmp)];
       H_bar = [H_bar ; H(:,:,c_tmp)];
    end
end