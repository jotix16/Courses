% This function performs systematic re-sampling
% Inputs:   
%           S_bar(t):       4XM
% Outputs:
%           S(t):           4XM
function newParticles = systematic_resample(Particles)
	
    global M % number of particles 
   
    % YOUR IMPLEMENTATION
    newParticles = Particles;
    cdf = cumsum(Particles.W);
    weight = 1/M;
    r = rand*weight;

    for m = 1:M
        j = find(cdf >= r + (m-1)/M, 1, 'first');
        newParticles.X(:,m) = Particles.X(:,j);
        newParticles.W(m) = weight;
        newParticles.landmarks{m} = Particles.landmarks{j};
    end
end