function [mu_means,cov_means] = get_statistics(Particles)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
%%
% nr of particles
M=length(Particles.X);

% nr of landmarks
N = length(Particles.landmarks{1}.mu);


% landmarks estimate
mu_means = zeros(2,N);
cov_means = zeros(2,2*N);

for m=1:M
    for n=1:N
        if ~isempty(Particles.landmarks{m}.seen{n})
            mu_means(:,n) = mu_means(:,n) + Particles.landmarks{m}.mu{n};
            cov_means(:,2*n-1:2*n) = cov_means(:,2*n-1:2*n) + Particles.landmarks{m}.cov{n};
        end
    end
end

mu_means = mu_means/M;
cov_means = cov_means/M;
end