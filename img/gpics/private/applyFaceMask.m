function [largePics, rim] = applyFaceMask(pics, graymask, picSize, erode1, erode2)
    for n = 1:numel(pics)
        tmp = pics{n};
        tmp(~graymask) = 127;
        template = zeros(picSize, picSize);
        tdims = size(template);
        dims = size(tmp);
        margins = tdims/2 - dims/2;
        tmp2 = template + 127;
        tmp2(margins(1)+1 : (margins(1)+dims(1)), margins(2)+1 : (margins(2)+dims(2))) = tmp;
        largePics{n} = tmp2;
    end

gtemplate = template;
gtemplate(margins(1)+1 : (margins(1)+dims(1)), margins(2)+1 : (margins(2)+dims(2))) = graymask;

c = imerode(gtemplate, true(erode1));
d = imerode(gtemplate, true(erode2));
rim = gtemplate + c + d;
end