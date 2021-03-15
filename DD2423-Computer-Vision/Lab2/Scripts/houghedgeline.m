function [linepar, acc] = houghedgeline(subsample, pic, scale, gradmagnthreshold, nrho, ntheta, nlines, verbose)
% close all;
%% Output
% linepar:              a list of (ρ, θ) parameters for each line segment,
% acc:                  the accumulator matrix of the Hough transform,
%% Input
% pic:                  the grey-level image,
% scale:                the scale at which edges are detected,
% gradmagnthreshold:    the threshold of the gradient magnitude.
% nrho:                 the number of accumulators in the ρ direction,
% nthetais:             the number of accumulators in the θ direction,
% threshold:            the lowest value allowed for the given magnitude,
% nlines: is            the number of lines to be extracted,
% verbose:              the degree of extra information and figures that will be shown.


%% Example 
%       houghedgeline(testimage2, 5, 100, 300, 300, 5, 1);
%       houghedgeline(testimage2, 5, 100, 500, 500, 10, 2)
%%
curves = extractedge(pic, scale, gradmagnthreshold, 'same');
magnitude = Lv(pic, 'same');
[linepar, acc] = houghline(subsample, curves, magnitude, nrho, ntheta, 12, nlines, 0);



% overlaycurves(magnitude, curves)
%% Verbose 
if verbose == 1
    figure
    overlaycurves(pic, linepar)
    axis([0 size(pic,1),0 size(pic,2)])
    title("Hough Lines")
else
    if verbose == 2
        subplot(1,2,1)
        showgrey(acc)
        l_1 = size(acc,1);
        l_2 = size(acc,2);
%         hold on
%         plot([l_1/2 l_1/2 l_1/2],[2,200,l_2], 'w')
%         plot([2,200,l_1], [l_2/2 l_2/2 l_2/2], 'w')
        title("Hough Transform")
        axis on
        ylabel('\rho','FontSize', 10) 
        xlabel('\theta/\pi','FontSize', 10)
        
        % Set axis of pi and theta
        xticklabels = -0.5:0.5:0.5;
        xticks = linspace(1, l_2, numel(xticklabels));
        set(gca, 'XTick', xticks, 'XTickLabel', xticklabels)
        
        yticklabels = -round(hypot(255,255)):120:round(hypot(255,255));
        yticks = linspace(1, l_1, numel(yticklabels));
        set(gca, 'YTick', yticks, 'YTickLabel', flipud(yticklabels(:)))
        
        subplot(1,2,2)
        overlaycurves(pic, linepar)
        axis([0 size(pic,1),0 size(pic,2)])
        title("Hough Lines")
    end
end