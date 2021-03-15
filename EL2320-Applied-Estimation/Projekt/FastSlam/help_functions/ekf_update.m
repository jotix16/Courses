function [weight,landmarks_k] = ekf_update(landmarks_k, x_k, z, c)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%EKF_UPDATE 
%Updates landmarks and finds weight of particle k using all measurements z_t.
%%   INPUT:
%           landmarks_k       old landmarks parameters of particle k
%           x_k               position of particle k [3x1]
%           z_t               measurements at t      [2xN] where N is nr of meassurements
%           c                 correspodence of measurements
%%   OUTPUT
%           weights           updated weight for particle k needed for resampling
%           landmarks_k       cell of updated landmarks for particle k (land.mu, land.cov,land.seen)
%           sigma             new covariance of landmark's position
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
global Q 
p0 = 0.5; % default importance weight


N = size(z,2);
% weights_from_meassurements = zeros(1,N);
weight = 1;

for n = 1:N
    %j = c(n); % landmark identity
    j = associate(x_k,landmarks_k,z(:,n));
    % landmark never seen before
    if (length(landmarks_k.seen) < j || isempty(landmarks_k.seen{j}) )
        % Update landmark parameters
        landmarks_k.nr = landmarks_k.nr + 1;
        landmarks_k.mu{j} = x_k(1:2) + z(1,n) .* [cos(z(2,n)+x_k(3)) ; sin(z(2,n)+x_k(3))];
        [H,~] = jacobian_observation_model(x_k, landmarks_k.mu{j});
        landmarks_k.cov{j}= H \ Q /  H';
        landmarks_k.seen{j} = true;
%         landmarks_k = land;
        weight = weight * p0;
    else
        [mu_j,cov_j,w] = EKF_update_landmark(x_k,landmarks_k.mu{j}, landmarks_k.cov{j},z(:,n));
        if w < 0.5
            continue
        end
        landmarks_k.mu{j} = mu_j;
        landmarks_k.cov{j}= cov_j;
        landmarks_k.seen{j} = true;
        weight = weight * w;
    end
end

end

