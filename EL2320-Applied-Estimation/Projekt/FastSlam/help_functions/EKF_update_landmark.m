function [mu_j,cov_j,w] = EKF_update_landmark(x_k,mu_kj,cov_kj,z)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%EKF_UPDATE_LANDMARK
%EKF update for a landmark with corresponding measurement z from possition x_k.
%%   INPUT:
%           x_k              [3x1]  position of particle k 
%           mu_kj            [2x1]  mean of landmark j of particle k at t-1              
%           cov_kj           [2x2]  covariance of landmark j of particle k at t-1           
%           z                [2x1]  measurements at t      
%%   OUTPUT
%           mu_j              new mean of jth landmark poistion of particle k
%           cov_j             new covariance of jth landmark poistion of particle k
%           w                 weight comming from meassurement z for resampling
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

global Q

[H, z_hat] = jacobian_observation_model(x_k, mu_kj);

Q_temp = H * cov_kj * H' + Q;
K = cov_kj * H' /Q_temp;


% compute inovation
nu = z-z_hat;
nu(2) = mod(nu(2)+pi,2*pi) - pi;

% new mean and cov
mu_j = mu_kj + K *nu;
cov_j = ( eye(2) - K*H )*cov_kj;

% weight from this measurement
eta = 1/(2*pi*det(Q_temp)^0.5); % mahalanobis factor
w = eta * exp(-0.5 * (nu' / Q_temp) * nu); 


end

