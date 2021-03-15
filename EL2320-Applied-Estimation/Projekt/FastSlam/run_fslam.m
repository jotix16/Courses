function [Particles, measurement_info]  = run_fslam(Particles, v, omega, delta_t, z, association_ground_truth)
%RUN_FSLAM Summary of this function goes here
%   Detailed explanation goes here

%
% set simulation mode
global DATA_ASSOCIATION % use association ground truth or perform ML data association
global RESAMPLE_MODE % re-sampling strategy for weighted particles

% import global variables
global landmark_ids % unique 
global t % global time
global M
% number of measurements available
n_measurements = size(z, 2); 

% 1) Predict step including particle diffusion
[Particles.X] = predict(Particles.X, v, omega, delta_t);

% check if measurements are available
if n_measurements > 0
        
    for m = 1:M
        c = 1;
        [w, Particles.landmarks{m}] = ekf_update(Particles.landmarks{m}, Particles.X(:,m), z, c);
        Particles.W(m) = Particles.W(m)*w;
    end
    Particles.W = Particles.W / sum(Particles.W);    
    
    % resample particles
    switch RESAMPLE_MODE
        case 1
            Particles = multinomial_resample(Particles);
        case 2
            Particles = systematic_resample(Particles);
    end
    
    
    measurement_info = [];
else
% there's no measurements available
% do nothing
    measurement_info = [];
end
%% Algorithm
% for each particle
    % PREDICT using motion model
    
    % for each measurements
        % initialize new landmark if not seen before
        % ekf_step for the landmark if seen before (GET WEIGHTS TOO)
        
    % get produkt of weights is the new weight for this particle
% normalize weights of particles

% RESAMPLE STEP
end

