% This function performs the maximum likelihood association and outlier detection given a single measurement.
% Note that the bearing error lies in the interval [-pi,pi)
%           mu_bar(t)           3X1
%           sigma_bar(t)        3X3
%           z_i(t)              2X1
% Outputs: 
%           c(t)                1X1
%           outlier             1X1
%           nu^i(t)             2XN
%           S^i(t)              2X2XN
%           H^i(t)              2X3XN
function [c, outlier, nu, S, H] = associate(mu_bar, sigma_bar, z_i)

    % Import global variables
    global Q % measurement covariance matrix | 2X2
    global lambda_m % outlier detection threshold on mahalanobis distance | 1X1
    global map % map | 2Xn
    
    % YOUR IMPLEMENTATION %
    [n,m] = size(map);
    H = zeros(2,3,m);
    S = zeros(2,2,m);
    nu = zeros(2,m);
    v = zeros(2,1);
    d = -1;
    c = 0;
    prob = -1;
    prob_tmp = 0;
    outlier = false;
    
    
    for j=1:m
        mu_bar(3) = mod(mu_bar(3) + pi , 2*pi) - pi;
        zhat = observation_model(mu_bar,j);
        H(:,:,j) = jacobian_observation_model(mu_bar, j, zhat);
        S(:,:,j) = H(:,:,j)  * sigma_bar * H(:,:,j)'  + Q;
        v(1) = z_i(1) - zhat(1);
        z_i(2) = mod(z_i(2) + pi , 2*pi) - pi;
        v(2) = mod(z_i(2) - zhat(2) + pi , 2*pi) - pi;   % make sure bearing error is calculated right
        nu(:,j) = v;
        dtemp = v' / S(:,:,j) * v; 
        prob_tmp = det(2 * pi * S(:,:,j)).^ -0.5 * exp(-0.5 * dtemp); % likelihood for corresponding to landmark j
        if  prob_tmp >= prob  % update associate by maximising the likelihood
            prob = prob_tmp;
            d = dtemp;
            c = j;
        end
    end 
           
    if d > lambda_m
        outlier = true;
    end
end