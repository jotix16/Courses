function [tau, initmap] = createTau2(MapIds, simSteps, odo,  finish,Q)
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
initmap=zeros(2,size(MapIds,2));
if finish >numPoses
    finish=numPoses;
end

for j = 1:numLandmarks
    land=MapIds(j);    
    tau_j = [];
    for t = 1:finish
        if simSteps{t}.containsLandmark(land)
            tau_j = [tau_j t]; %tau_j holds timesteps where j_th landmark was seen
        end
    end
    tau{j} = tau_j;
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


