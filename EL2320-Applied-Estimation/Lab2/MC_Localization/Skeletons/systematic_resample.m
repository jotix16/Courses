% This function performs systematic re-sampling
% Inputs:   
%           S_bar(t):       4XM
% Outputs:
%           S(t):           4XM
function S = systematic_resample(S_bar)
	
    global M % number of particles 
    
    % YOUR IMPLEMENTATION
    S = zeros(4,M);
    cdf = cumsum(S_bar(4,:));
    weight = 1/M;
    
    r = rand*weight;

    for m = 1:M
        j = find(cdf >= r + (m-1)/M, 1, 'first');
        S(:,m) = [S_bar(1:3,j); weight];
    end
end