% This function performs multinomial re-sampling
% Inputs:   
%           S_bar(t):       4XM
% Outputs:
%           S(t):           4XM
function S = multinomial_resample(S_bar)

    global M % number of particles
    
    % YOUR IMPLEMENTATION
    S = zeros(4,M);
    cdf = cumsum(S_bar(4,:));
    weight = 1/M;
    
    for m = 1:M
        r = rand;
        j = find(cdf >= r, 1, 'first');
        S(:,m) = S_bar(:,j);
        S(4,m) = weight;
    end
end
