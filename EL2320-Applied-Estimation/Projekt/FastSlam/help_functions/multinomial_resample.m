% This function performs multinomial re-sampling
% Inputs:   
%           S_bar(t):       4XM
% Outputs:
%           S(t):           4XM
function newParticles = multinomial_resample(Particles)

    global M % number of particles

    % YOUR IMPLEMENTATION
    newParticles = Particles;
    cdf = cumsum(Particles.W);
    weight = 1/M;
    for m = 1:M
        r = rand;
        j = find(cdf >= r, 1, 'first');
        newParticles.X(:,m) = Particles.X(:,j);
        newParticles.W(m) = weight;
        newParticles.landmarks{m} = Particles.landmarks{j};
    end
end
