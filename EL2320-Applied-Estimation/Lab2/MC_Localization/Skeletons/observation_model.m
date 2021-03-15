% This function is the implementation of the measurement model.
% The bearing should be in the interval [-pi,pi)
% Inputs:
%           S(t)                4XM
%           j                   1X1
% Outputs:  
%           z_j                 2XM   Observation to jth point of map

function z_j = observation_model(S, j)

    global map % map including the coordinates of all landmarks | shape 2Xn for n landmarks
    global M % number of particles

    % YOUR IMPLEMENTATION
    WM=repmat(map(:,j),1,M);
    delta = WM - S(1:2,:);
%     delta = map(:,j) - S(1:2,:);

    z_j = [ hypot(delta(1,:), delta(2,:)); mod( atan2(delta(2,:), delta(1,:)) - S(3,:) + pi, 2*pi) - pi  ];

end