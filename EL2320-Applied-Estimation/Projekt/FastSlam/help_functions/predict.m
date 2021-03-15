% This function performs the prediction step.
% Inputs:
%           S(t-1)            4XN
%           v                 1X1
%           omega             1X1
% Outputs:   
%           S_bar(t)          4XN
function [X_bar] = predict(X, v, omega, delta_t)

    % Comment out any S_bar(3, :) = mod(S_bar(3,:)+pi,2*pi) - pi before
    % running the test script as it will cause a conflict with the test
    % function. If your function passes, uncomment again for the
    % simulation.

    global R % covariance matrix of motion model | shape 3X3
    global M % number of particles
    
    % YOUR IMPLEMENTATION
    u= [ v * delta_t * cos(X(3,:)); v * delta_t * sin(X(3,:)) ];
    u_3 = omega * delta_t;
    X_bar(1:2,:) =  X(1:2,:) + u;
    X_bar(3,:) = X(3,:) + u_3;
    X_bar(1:3,:)=X_bar(1:3,:)+randn(3,M) .* repmat(sqrt(diag(R)),1,M);
    X_bar(3, :) = mod(X_bar(3,:)+pi,2*pi) - pi;
end