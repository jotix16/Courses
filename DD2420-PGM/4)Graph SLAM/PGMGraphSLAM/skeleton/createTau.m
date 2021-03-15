function [tau, initmap, outliers,simSteps] = createTau(MapIds, simSteps, odo,  finish,finish2,muMap,Q)
%  make a list for each landmark in MapIs of iterations that the landmark 
%  was seen. 
%       MapIds: 1 x M   list of id's for the landmarks
%       simSteps: a cellarray of SimStep objects summarizing the
%       measurements for each iteration
%       odo:  2xN  the ds and dtheta of the robot pose
%       finish:  the step to stop collecting tau  
%  Output
%       tau a cell array of the lists.
numPoses = size(simSteps,2);
numLandmarks = size(MapIds,2);

tau = cell(1,numLandmarks);
outliers=cell(1,numLandmarks);
% initmap=zeros(2,size(MapIds,2));
initmap = muMap;
if finish >numPoses
    finish=numPoses;
end

for j = 1:numLandmarks
    land=MapIds(j);
    %%
    land = j;
    tau_j = [];
    outliers_j = [];
    for t = 1:finish
        if simSteps{t}.containsLandmark(land)
            %%%
            %%%%%%%%%%%%   
            index=find(simSteps{t}.seenLandmarks(3,:)==land);
                z = simSteps{t}.getLandmark(index);
            if all(initmap(1:2,land)~=[0;0]) && t<finish2
                z=z(1:2);
                %What do we expect to measure for this landmark from mu
                zBar = observation_model(simSteps{t}.odometry, initmap, land);
                % Compute innovation
                innovation = z - zBar;
                innovation(2,1) = mod(innovation(2,1) + pi, 2*pi) - pi;   
                % Keep values of yaw within limits
                %% HERE MUST COME OUTLIER REJECTION
                eta = 1; 
%                 eta = 1/(2*pi*det(Q)^0.5); % mahalanobis factor
                prob = eta * exp(-0.5 * (innovation' * inv(Q) * innovation)); % psi_tmp: [M*N*n_m,1]
                if prob < 0.2
                    simSteps{t}.seenLandmarks(:,index)=[];
                    outliers_j = [outliers_j t];
                    continue
                end
            end
            %%%%%%%%%%%%%
            %%%
%             if z(1)>20 && t<finish2
%                 outliers_j = [outliers_j t];
%                 continue
%             end
            tau_j = [tau_j t]; %tau_j holds timesteps where j_th landmark was seen
        end
    end
    % predict position of landmark as seen for first time
    tau{j} = tau_j;
    outliers{j}=outliers_j;
    if size(tau_j)>0
        odom=simSteps{tau_j(1)}.odometry;
        % get the index where  landmark land = MapIds(j) is seen for first time
        index=find(simSteps{tau_j(1)}.seenLandmarks(3,:)==land);
        % get the measurement from timestep where landmark land = MapIds(j) is seen for first time
        r=simSteps{tau_j(1)}.getLandmarkRange(index);
        b=simSteps{tau_j(1)}.getLandmarkBearing(index);
        % predict the position of landmark
        initmap(1:2,j)=[odom(1)+r*cos(odom(3)+b); odom(2)+r*sin(odom(3)+b)];
    end
end
end


        %%%%%%%%%%%%
   