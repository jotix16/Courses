% This function is the implementation of the measurement model.
% The bearing should be in the interval [-pi,pi)
% Inputs:
%           x(t)                           3X1
%           j                              1X1
% Outputs:  
%           h                              2X1
function z_j = observation_model(x, j)

    global map % map | 2Xn for n landmarks

    % YOUR IMPLEMENTATION %
    x(3) = mod(x(3) + pi , 2*pi) - pi;

    delta = map(:,j) - x(1:2);
    z_j = [ hypot(delta(1), delta(2)); mod( atan2(delta(2), delta(1)) - x(3) + pi, 2*pi) - pi  ];
    
end