
function [] = get_composite_gender_ml(rawData, rppath, vp_selection, patches, npic, outFileName, conCodes)
 
%n           = 0 ;                     % Stored or entered by user.
%fid = fopen([rppath, outFileName], 'w'); 
% select participants
tmpcell = {rawData(:).vpcode}';
if all(ismember(vp_selection, 'all'))
    vpnums = 1:numel(rawData);
else
    vpnums = find(ismember(tmpcell, vp_selection));
end
% loop
max_alpha = max(patches{1}(:));

for vpnr = 1:numel(vpnums)
vp = vpnums(vpnr);

%rawData(vp).b_centers{1,1} % 

for trial = 1:numel(rawData(vp).b_centers)
for scale = 1:5
%n = n + 1;

%bdims = rawData(vp).b_dims{trial}{scale};
%bdims(:,2) = size(patches{scale},1);
%bdims(:,4) = size(patches{scale},1);
[~, bubble_dims, face_coords]      = prepare_alpha_ml(patches{scale}, npic{scale}, rawData(vp).b_centers{trial}{scale}, rawData(vp).facedims);
rawData(vp).b_dims{trial}{scale}   = bubble_dims;
rawData(vp).f_coords{trial}{scale} = face_coords;
rp                 = add_alphaplane(patches{scale}, npic{1}, rawData(vp).b_centers{trial}{scale}, ...
                     rawData(vp).b_dims{trial}{scale}, rawData(vp).f_coords{trial}{scale}); %one patch size at a time
rp(rp > max_alpha) = max_alpha;
a_planes{scale}    = rp;
end

% 1:20 == 21:40 == 41:60
if ismember(rawData(vp).outmat(trial,3), 1:20)
    picNumber = rawData(vp).outmat(trial,3);    
elseif ismember(rawData(vp).outmat(trial,3), 21:40)
    picNumber = rawData(vp).outmat(trial,3) - 20;
elseif ismember(rawData(vp).outmat(trial,3), 41:60)
    picNumber = rawData(vp).outmat(trial,3) - 40;
end

[t_stim, ~] = sum_to_target_ml(a_planes, npic(picNumber,:));
nPadded     = sprintf( '%04d', rawData(vp).outmat(trial, 1));
if rawData(vp).outmat(trial, 6) == 1 % response correct
    imwrite(t_stim/255, [rppath, conCodes{rawData(vp).outmat(trial, 8)}, ...
        '/correct/comp-', rawData(vp).vpcode, '-', num2str(scale), '-', nPadded, '.png'], 'PNG');
elseif rawData(vp).outmat(trial, 6) == 0 % response incorrect
    imwrite(t_stim/255, [rppath, conCodes{rawData(vp).outmat(trial, 8)}, ...
        '/incorrect/comp-', rawData(vp).vpcode, '-', num2str(scale), '-', nPadded, '.png'], 'PNG');
end
clear t_stim
% 1 = female, 2 = male

%picText = ['f', num2str(picNumber + 1000)];
% add 1000, 2000 etc for 
%fprintf(fid, '%s\t%s\t%s\t%s\t%s\t%s\n', ['comp', nPadded], rawData(vp).vpcode, num2str(scale), num2str(rawData(vp).outmat(trial,17)), rawData(vp).group, picText);
end
end
%fclose(fid);
end


