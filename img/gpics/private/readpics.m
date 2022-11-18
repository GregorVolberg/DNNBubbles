function [pics, allpics] = readpics(dirs, filepattern)
    allpics = [];
    for ndir = 1:numel(dirs)
        stimnames = dir([dirs{ndir}, filepattern]);
        fullnames = strcat(dirs{ndir}, {stimnames.name});
        allpics   = [allpics, fullnames];
    end
    
    for k = 1:numel(allpics)
    pics{k} = double(imread(allpics{k}));
    end
end
