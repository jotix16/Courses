% This function performs the ML data association
%           S_bar(t)                 4XM
%           z(t)                     2Xn
%           association_ground_truth 1Xn | ground truth landmark ID for
%           every measurement
% Outputs: 
%           outlier                  1Xn
%           Psi(t)                   nXMx1
function [outlier, Psi, c] = associate(S_bar, z, association_ground_truth)
    if nargin < 3
        association_ground_truth = [];
    end

    global DATA_ASSOCIATION % whether to perform data association or use ground truth
    global lambda_psi % threshold on average likelihood for outlier detection
    global Q % covariance matrix of the measurement model
    global M % number of particles
    global N % number of landmarks
    global landmark_ids % unique landmark IDs
    
    % YOUR IMPLEMENTATION
    n = size(z,2);
    outlier = zeros(1,n);
    Psi = zeros(n,M,1);
    c = zeros(n,M);
    
    for k = 1:N
       Z(:,:,k) = observation_model(S_bar, k); 
    end   

    
    
    for i = 1:n % For all measurements
        not_outlier = false;
        for m = 1:M % For all particles
            prob = -1; 
            for k = 1:N % For all landmarks(map)
                v(1,1) = z(1,i) - Z(1,m,k);
                v(2,1) = mod(z(2,i) - Z(2,m,k) + pi , 2*pi) - pi; % Inovation of measurement (z_i) to d_m_k)
                                                                  % which is the distance from particle m to landmark k.
                dtemp = v' / Q * v; 
                prob_tmp = 1/(2*pi) * det( Q).^ -0.5 * exp(-0.5*dtemp); % likelihood for d_m_k corresponding to landmark k
                if  prob_tmp > prob  % update associate by maximising the likelihood
                    prob = prob_tmp;
                    c(i,m) = k;
                end
            end
            Psi(i,m,1) = prob;
            outlier(1,i) = ~not_outlier;
        end
    end    
    % Detect outlier
    outlier = (sum(Psi(:,:,1),2)/M) < lambda_psi;
    
    
    