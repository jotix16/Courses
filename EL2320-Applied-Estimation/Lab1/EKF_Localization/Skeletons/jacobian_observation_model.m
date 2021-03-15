 % This function is the implementation of the jacobian measurement model
% required for the update of the covariance function after incorporating
% the measurements
% Inputs:
%           x        3X1
%           mu_j         2X1
% Outputs:  
%           H           2X3
function H = jacobian_observation_model(x, mu_j)

    % YOUR IMPLEMENTATION %
    H = [ (x(1)-map(1,j))/z_j(1) (x(2)-map(2,j))/z_j(1) 0; -(x(2)-map(2,j))/z_j(1)^2 (x(1)-map(1,j))/z_j(1)^2 -1];
    
end
