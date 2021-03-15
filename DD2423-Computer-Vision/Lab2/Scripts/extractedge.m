function curves = extractedge(inpic,scale, threshold, shape)
%extractedge(inpic,scale, threshold, shape) 
%-------------------------------------------
%Extracts edges using first and second order derivatives!
%The output has the form of curves as in "help contourc".
    
    % Thresholded magnitude of first derivative   |\/L| == magnitude
    Lvmask = (Lv(discgaussfft(inpic, scale), shape)- threshold > 0) -0.5;
    
    % Magnitude of second derivative on each pixel
    Lvv = Lvvtilde(discgaussfft(inpic, scale ), 'same');
    
    % Negative part of magnitude of third order derivative in each pixel
    Lvvvmask = (Lvvvtilde(discgaussfft(inpic, scale ), 'same') < 0) +0.5 ;
    
    curves = zerocrosscurves(Lvv, Lvmask);
    curves = thresholdcurves(curves, Lvvvmask);
end

