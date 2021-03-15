
function c = associate(x_k,landmarks_k, z, association_ground_truth)
    if nargin == 4
        outlier = 0;
        Psi = 0;
        c = association_ground_truth;
        return
    end
    
    if nargin < 3
        association_ground_truth = [];
    end

    global lambda_psi % threshold on average likelihood for outlier detection
    global Q

    n_measurements = size(z, 2); % number of measurements
    N = landmarks_k.nr;
    psi = zeros(1, N); % predicted measurements

    if N == 0
        c = 1;
        return
    end
    
    % get predicted measurements for all landmarks
    for j = 1:N
        [H, z_hat] = jacobian_observation_model(x_k, landmarks_k.mu{j});
        nu = z_hat - z;
        nu(2) = mod(nu(2)+pi,2*pi) - pi;
        Q_temp = H * landmarks_k.cov{j} * H' + Q;
        eta = 1/(2*pi*det(Q_temp)^0.5); % mahalanobis factor
        psi(1,j) = eta * exp(-0.5 * (nu' / Q_temp) * nu);  
    end


    % get maximum likelihood and associated index
    [psi_max, c] = max(psi);
    

    if psi_max<= lambda_psi
        c = N+1;
    end
end
