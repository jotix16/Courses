 % This function is the implementation of the jacobian measurement model
% required for the update of the covariance function after incorporating
% the measurements
% Inputs:
%           x(t)        3X1
%           j           1X1 which landmark (map column)
%           z_j         2X1
% Outputs:  
%           H           2X2 Jacobi matrix at mu_j
function [H, z_bar] = jacobian_observation_model(x_k, mu_j)
    
    % calculate the measurement prediction
    z_bar = observation_model(x_k, mu_j);
    
    % calculate the Jacobi matrix
    delta_temp = mu_j-x_k(1:2);
    
    H = [ delta_temp(1)/z_bar(1), delta_temp(2)/z_bar(1); -delta_temp(2)/z_bar(1)^2, delta_temp(1)/z_bar(1)^2  ];
        
end
