% This function is the implementation of the measurement model.
% The bearing should be in the interval [-pi,pi)
% Inputs:
%           x_k                 2X1 Coordinates of particle k 
%           mu_j                2X1 Mean of landmark j
% Outputs:  
%           z_j                 2X1 Observation to jth landmark from x_k

function z_j = observation_model(x_k, mu_j)


    % YOUR IMPLEMENTATION
    delta =  mu_j - x_k(1:2);
    z_j = [ hypot(delta(1), delta(2)); mod( atan2(delta(2), delta(1)) - x_k(3) + pi, 2*pi) - pi  ];

end