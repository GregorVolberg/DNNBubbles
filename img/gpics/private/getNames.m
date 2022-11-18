function [pathname, foldernames, filenames] = getNames(filepath)
[tmp] = dir(filepath);
pathname = tmp(1).folder;
folders = {tmp.name};
isdir   = [tmp.isdir];
foldernames = folders(isdir)';
foldernames = foldernames(3:length(foldernames));
for j = 1:numel(foldernames)
    fullpath = [pathname, '\', foldernames{j}, '\'];
    tmp2 = dir([fullpath, '*.mp4']);
    filenames{j} = {tmp2.name}';
end
end
