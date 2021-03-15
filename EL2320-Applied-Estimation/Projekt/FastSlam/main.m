    close all
    mapfile = 'map4.txt';
    datafile = 'dataset4.txt';
    dataset_id = 4;

    %% initialize Params
    global t % global simulation time
    global M % number of particles
    t = 0;
    
    global R % covariance matrix of the motion model
    global Q % covariance matrix of the measurement model
    global lambda_psi % threshold on average likelihood for outlier detection

    % default parameters for chosen dataset for reset button in app
    global default_R 
    global default_Q 
    global default_lambda_psi % threshold on average likelihood for outlier detection

    
    % Dataset 1
    if dataset_id == 1
        default_R = [0.01^2, 0, 0; 0, 0.01^2, 0; 0, 0, 0.01^2];
        default_Q = [0.01^2, 0; 0, 0.01^2];
        default_lambda_psi = 2;
        M = 50;
    % Dataset 2
    elseif dataset_id == 2
        default_R = [0.003, 0, 0; 0, 0.003, 0; 0, 0, 0.003];
        default_Q = [0.01, 0; 0, 0.01];
        default_lambda_psi = 2;
        M = 50;
    % Dataset 3
    elseif dataset_id == 3
        default_R = [0.00051, 0, 0; 0, 0.00051, 0; 0, 0, 0.00081];
        default_Q = [0.0001, 0; 0, 0.0001];
        default_lambda_psi = 2;
        M = 70;
    % Dataset 4
    elseif dataset_id == 4
        default_R = [0.00071, 0, 0; 0, 0.00071, 0; 0, 0, 0.00081];
        default_Q = [0.0001, 0; 0, 0.0001];
        default_lambda_psi = 0.05;
        M = 70;
    end
    % set global parameters to default values
    R = default_R;
    Q = default_Q;
    lambda_psi = default_lambda_psi;
    
    
    
    %% initialize and draw map
    map_data = load(['DataSets/' mapfile]);
    global map
    global landmark_ids
    global N
    map = map_data(:,2:3)'; % map including the coordinates of all landmarks | shape 2Xn for n landmarks
    landmark_ids = map_data(:,1)'; % contains the ID of each landmark | shape 1Xn for n landmarks
    N = length(landmark_ids);
    

    
    
    %% initialize simulation mode
    global RESAMPLE_MODE % use ground-truth data instead of ML data association
    global DATA_ASSOCIATION % perform batch update instead of sequential update
    
    % set default values
    DATA_ASSOCIATION = false; 
    RESAMPLE_MODE = 2;
    
    %% RUN SIMULATION
    run_simulation(datafile)
