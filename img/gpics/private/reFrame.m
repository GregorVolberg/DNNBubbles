function [reframed, picsize, margins] = reFrame(smoothedPics, framesize)    
    template = zeros(round(size(smoothedPics{1})*framesize));
    tdims = size(template);
    dims = size(smoothedPics{1});
    margins = round(tdims/2 - dims/2);
    for k =1:numel(smoothedPics)
    tmp2 = template + 127;
    tmp2(margins(1)+1 : (margins(1)+dims(1)), margins(2)+1 : (margins(2)+dims(2))) = smoothedPics{k};
    reframed{k} = tmp2;
    end
    picsize = size(reframed{1});
    margins = [margins(1)+1, (margins(1)+dims(1)), margins(2)+1, (margins(2)+dims(2))];
end
