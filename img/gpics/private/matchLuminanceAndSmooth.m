function [smoothedPics, picdims] = matchLuminanceAndSmooth(largePics, rim, targetLum, smoothkernel)
    
    % call SHINE toolbox for luminance machting; 
    % reduces faceMask by 10 pixels befor luminance matching in order to account for different face sizes 
    lumscaled = lumMatch(largePics, rim > 1, targetLum); 
    
    % compute final picture dimension
    [a,b]=ind2sub(size(rim), find(rim > 1));
    upperleft  = [min(a), min(b)];
    lowerright = [max(a), max(b)];
    
    % smooth between rim2 and face area
    for n = 1:numel(largePics)
        tgt = lumscaled{n};
        tgt(rim <=1) = 127; % sets outer rim intensities to 127 = gray
        innerrim = tgt; innerrim(rim < 3) = 0; % cuts out inner rim (the face)
        tt=imgaussfilt(tgt, smoothkernel);     % smoothes target picture (but not face)
        tt(rim ==3) = innerrim(rim==3);        % copies face back into target picture
        smoothedPics{n} = tt(upperleft(1):lowerright(1), upperleft(2):lowerright(2));
        
    end
    
    % also return picture dimensions (width, height)
    picdims = size(smoothedPics{1});
end