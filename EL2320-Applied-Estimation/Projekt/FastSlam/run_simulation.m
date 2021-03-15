function [] = run_simulation(simulationfile)
%RUN_SIMULATION Summary of this function goes here
%   Detailed explanation goes here

    %% Initialize simulation data
    % store simulation data in cell array
    fid = fopen(['DataSets/' simulationfile], 'r');
    if fid <= 0
      fprintf('Failed to open simoutput file "%s"\n\n', simulationfile);
      return
    end
    simulation_data = {};
    while 1
        line = fgetl(fid);
        if ~ischar(line)
            break
        end
        simulation_data = [simulation_data(:)' {line}];
    end
    fclose(fid);

    %% Initialize particles
    global M % number of particles
    global start_pose  % start pose in case of tracking problem
    
    % tracking problem with initial estimate
    initial_data = sscanf(simulation_data{1}, '%f');
    start_pose = initial_data(7:9);
    % the particle set
    Particles.X = repmat(start_pose, 1, M);
    Particles.W = 1 / M * ones(1, M);
    Particles.landmarks_nr = zeros(1, M);
    Particles.landmarks = cell(1,M);
    land.nr = 0;
    land.mu = {};
    land.cov = {};
    land.seen = {};
    land.counter = {};
    Particles.landmarks(1,:) = {land};
    % Add landmark j for particle m:   Particles.landmarks{m}.mu{j} =land
    % where land.mu = [x;y];
    %       land.cov= ones(2);
    %       land.seen = true;
    
    % Check if landmark j not seen from particle m: if (length(Particles.landmarks{m}.seen) <j || isempty(land.seen{j}) )
    
    %% Initialize parameters and data structures
    % initial wheel-encoder readings
    enc = [0; 0]; 
    
    % odometry parameters
    E_T = 2048; % encoder ticks per wheel evolution
    B= 0.35; % distance between contact points of wheels in m
    R_L = 0.1; % radius of the left wheel in m
    R_R = 0.1; % radius of the right wheel in m

    % number of timesteps in simulation data
    n_timesteps = size(simulation_data, 2);
    
    % import global variables
    global t % global time
    t = 0;
    
    nloops = 1;
    % save simulation statistics
    poses = zeros(2, n_timesteps*nloops);
    true_poses = zeros(2, n_timesteps*nloops);
    odom_poses = zeros(2, n_timesteps*nloops);
    pose_errors = zeros(3, n_timesteps*nloops);
    sigmas = zeros(9, n_timesteps*nloops);

    %% go through all measurement data and control input (& correspodences) 
    %% run simulation
loop_t = 0;
for i = 1:nloops
    for timestep = 1:n_timesteps
       
        % read data for current timestep
        line = simulation_data{timestep};
        timestep_data = sscanf(line, '%f');
        
        
        % save values from previous timestep
         
        pt = t; % previous time in seconds
        penc = enc; % wheel-encoder information of previous timestep

        % get information from simulationfile
        t = timestep_data(1)+ loop_t; % current time in seconds (add loops)
        if timestep == n_timesteps loop_t= t; end
        odom = timestep_data(2:4); % odometry information
        enc = timestep_data(5:6); % wheel-encoder information
        true_pose = timestep_data(7:9); % ground-truth pose  
        n_measurements = timestep_data(10); % number of observations available
        if (n_measurements > 0) % if observations for current timestep available
            bearings = timestep_data(12:3:end); % bearing of observation
            ranges = timestep_data(13:3:end); % distance to observed landmark
            z = [ranges'; bearings']; % measurements
            association_ground_truth = timestep_data(11:3:end);  % id of observed landmark
        else
            bearings = [];
            ranges = [];
            z = [ranges';bearings'];
            association_ground_truth = [];
        end

        % relative information to last timestep
        delta_t = t - pt; % time difference
        delta_enc = enc - penc; % wheel-encoder difference

        % compute odometry information
        [v, omega] = calculate_odometry(delta_enc(1), delta_enc(2), E_T, B, R_R, R_L, delta_t);
        
        %% run fslam_particle_filter
        [Particles, measurement_info] = run_fslam(Particles, v, omega, delta_t, z, association_ground_truth);

        %% get measurement statistics from FastSlam
        
        % get pose estimate
        mu = mean(Particles.X(1:2,:), 2);
        mu(3) = atan2(mean(sin(Particles.X(3, :))), mean(cos(Particles.X(3, :))));
        pos_sigma = cov(Particles.X(1:2,:)');
        var_theta = var(sin(Particles.X(3, :))) + var(cos(Particles.X(3, :)));
        % poses
        poses(:, (i-1)*n_timesteps + timestep) = mu(1:2);
        true_poses(:, (i-1)*n_timesteps + timestep) = true_pose(1:2);
        odom_poses(:, (i-1)*n_timesteps + timestep) = odom(1:2);
        
        
        sigma = zeros(3,3);
        sigma(1:2, 1:2) = pos_sigma;
        sigma(3, 3) = var_theta;
        sigmas(:,(i-1)*n_timesteps + timestep) = sigma(:);

        % compute pose error
        pose_error = true_pose - mu;
        pose_error(3) = mod(pose_error(3)+pi,2*pi)-pi;
        pose_errors(:, (i-1)*n_timesteps +  timestep) = pose_error;

    end
end  
    
    %% Vizualize Results of FastSlam
    visualize(Particles)
    hold on
    plot(poses(1,:),poses(2,:),'b')
    plot(true_poses(1,:),true_poses(2,:),'k')
    plot(odom_poses(1,:),odom_poses(2,:),'y')
     
    

    %% Error Plots
    get_error_plot = false
    if get_error_plot
     % get error statistics
        mex = mean(pose_errors(1,:));
        mey = mean(pose_errors(2,:));
        met = mean(pose_errors(3,:) / (2*pi) * 360);
        maex = mean(abs(pose_errors(1,:)));
        maey = mean(abs(pose_errors(2,:)));
        maet = mean(abs(pose_errors(3,:) / (2*pi) * 360));
        
        % plot errors and covariance
        figure('Name', 'Evolution State Estimation Errors');
        clf;
        subplot(3,1,1);
        plot(pose_errors(1,:));
        ylabel('error\_x [m]');
        title(sprintf('error on x, mean error = %.2fm, mean absolute err = %.2fm', mex, maex));
        subplot(3,1,2);
        plot(pose_errors(2,:));
        ylabel('error\_y [m]');
        title(sprintf('error on y, mean error = %.2fm, mean absolute err = %.2fm', mey, maey));
        subplot(3,1,3);
        plot(pose_errors(3,:) / (2*pi) * 360);
        xlabel('simulation time [s]');
        ylabel('error\_\theta [�]');
        title(sprintf('error on \\theta, mean error = %.2f^{\\circ}, mean absolute err = %.2f^{\\circ} ', met, maet));
        
        figure('Name', 'Evolution State Estimation Covariance Matrix');
        clf;
        subplot(3,1,1);
        plot(sigmas(1,:));
        title('\Sigma(1,1)');
        subplot(3,1,2);
        plot(sigmas(5,:));
        title('\Sigma(2,2)');
        subplot(3,1,3);
        plot(sigmas(9,:));
        title('\Sigma(3,3)');
    end  
end

