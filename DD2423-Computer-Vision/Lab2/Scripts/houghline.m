function [linepar, acc] = houghline(subsample, curves, magnitude, nrho, ntheta, threshold, nlines, verbose) 
%% Output
% linepar:      a list of (x,y) parameters for each line segment,
% acc:          the accumulator matrix of the Hough transform,
%% Input
% curves:       the polygons from which the transform is to be computed,
% magnitude:    an image with one intensity value per pixel
% nrho:         the number of accumulators in the ρ direction,
% nthetais:     the number of accumulators in the θ direction,
% threshold:    the lowest value allowed for the given magnitude,
% nlines: is    the number of lines to be extracted,
% verbose:      the degree of extra information and figures that will be shown.

%%
% Check if input appear to be valid
if nargin ~= 8
  error( 'Wrong # of arguments.')
  return
% elseif ~nrho==ntheta==nlines
  error('Bad arguments: nrho, ntheta, nlines must be equal.')
  return
end

%% Allocate spaces
acc = zeros(nrho, ntheta); % Allocate accumulator space
linepar = zeros(2,nlines*3); % Allocate Linearpar space
outcurves = zeros(2,nlines*3); % Space for the curves to verbose

%% Define a coordinate system in the accumulator space
%               y-axis
%             ------------
%   x-axis    |
%             |
%             |
%             |
% Thetas definitition
thetas = linspace(-pi/2, pi/2, ntheta);

% Rho definition
rho_max = hypot(255,255);
delta_rho = 2*rho_max/(nrho-1);
rho_shift = rho_max/delta_rho;  % used to find index of a certain rho
rhos = linspace(-rho_max, rho_max, nrho);


%% Loop over all the input curves 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
insize = size(curves, 2);
trypointer = 1;
numcurves = 0;
while trypointer <= insize
  polylength = curves(2, trypointer);
  numcurves = numcurves + 1;
  trypointer = trypointer + 1;

  for polyidx = 1:polylength
    x = curves(2, trypointer);
    y = curves(1, trypointer);
    trypointer = trypointer + 1;
    
    %   Update accumulator
    val = abs(magnitude(round(x),round(y)));
    if val > threshold
        for theta_index = 1:ntheta
            theta = thetas(theta_index);
            p = x*cos(theta) + y*sin(theta);
            rho_index = find(rhos < p, 1, 'last');
            % can find the respective rho_index in constant time too
%             acc(rho_index, theta_index) = acc(rho_index, theta_index) + 1;
%             acc(rho_index, theta_index) = acc(rho_index, theta_index) + log(val);
              acc(rho_index, theta_index) = acc(rho_index, theta_index) + val;
%             acc(rho_index, theta_index) = acc(rho_index, theta_index) + val^2;
%             acc(rho_index, theta_index) = acc(rho_index, theta_index) + val^3;
        end    
    end
  end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Extract local maxima from the accumulator
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Do smoothing of histogram to handle high resolution
if (subsample)
    acc = binsubsample(acc);
    thetas = linspace(-pi/2, pi/2, ntheta/2);
    rhos = linspace(-rho_max, rho_max, nrho/2);
end



[pos value] = locmax8(acc);
[dummy indexvector] = sort(value);
nmaxima = size(value, 1);


% Compute a line for each one of the strongest responses in the accumulator
for idx = 1:nlines
    thetaidxacc = pos(indexvector(nmaxima - idx + 1), 2);
    rhoidxacc  = pos(indexvector(nmaxima - idx + 1), 1);
    theta = thetas(thetaidxacc);
    rho = rhos(rhoidxacc);
    linepar(:,idx) = [rho; theta ];
    
    
    
    x0=0;
    y0=rho/sin(theta);
    dx=rho_max^2;
    dy= (rho-dx*cos(theta)) / sin(theta);
    % Compute a line for each one of the strongest responses in the accumulator
    outcurves(1,4*(idx-1) + 1) = 0; % not significant
    outcurves(2,4*(idx-1) + 1) = 3; 
    outcurves(2,4*(idx-1) + 2) = x0 - dx;
    outcurves(1,4*(idx-1) + 2) = y0 - dy;
    outcurves(2,4*(idx-1) + 3) = x0 ;
    outcurves(1,4*(idx-1) + 3) = y0 ;
    outcurves(2,4*(idx-1) + 4) = x0 + dx;
    outcurves(1,4*(idx-1) + 4) = y0 + dy;  
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

linepar = outcurves;

%% Verbose
if verbose == 1
    overlaycurves(magnitude, linepar)
    axis([0 size(magnitude,1),0 size(magnitude,2)])
    title("Hough Lines")
else
    if verbose == 2
        subplot(1,2,1)
        showgrey(acc)
        title("Hough Transform")
        subplot(1,2,2)
        overlaycurves(magnitude, linepar)
        axis([0 size(magnitude,1),0 size(magnitude,2)])
        title("Hough Lines")
    end
end
        
