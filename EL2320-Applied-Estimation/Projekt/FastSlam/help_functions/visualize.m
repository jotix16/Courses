function [] = visualize(Particles)
%UNTITLED4 Summary of this function goes here
%   Detailed explanation goes here



    %% Vizualize Results of FastSlam

    % include margin around landmarks
    global map
    margin = 10;
    xmin = min(map(1, :)) - margin;
    xmax = max(map(1, :)) + margin;
    ymin = min(map(2, :)) - margin;
    ymax = max(map(2, :)) + margin;

    % draw map landmarks
    plot(map(1, :), map(2, :), 'ko')
    hold on
    axis([xmin xmax ymin ymax])

    [mu_means,cov_means] = get_statistics(Particles);
    plot(mu_means(1,:),mu_means(2,:),"rx")
    for i=1:length(mu_means)
        if mu_means(:,i) == 0
            continue
        end
        p = get_uncertainty_ellipse(mu_means(:,i), cov_means(:,2*i-1:2*i));
        plot(p(1,:),p(2,:));
    end
end

