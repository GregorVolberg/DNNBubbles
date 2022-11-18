function [resampled, picsize] = reSamp(smoothedPics, fraction)

    for k = 1:numel(smoothedPics)
    resampled{k} = imresize(smoothedPics{k}, fraction, 'nearest');
    end

    picsize = size(resampled{1});
end