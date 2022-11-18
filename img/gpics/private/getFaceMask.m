function [graymask, avgpic] = getFaceMask(pics, minIntensity)
tmp = zeros(size(pics{1}));
for k = 1:numel(pics)
    tmp = tmp + pics{k};
end
avgpic = tmp / numel(pics);
zeromask =  avgpic < minIntensity;
graymask = zeromask == 0;
end