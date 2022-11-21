
function [] = get_composite_emotion_ml(rawData, rppath, vp_selection, patches, npic, conCodes)
 
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

for trial = 1:numel(rawData(vp).b_centers)
for scale = 1:5
[~, bubble_dims, face_coords]      = prepare_alpha_ml(patches{scale}, npic{scale}, rawData(vp).b_centers{trial}{scale}, rawData(vp).facedims);
rawData(vp).b_dims{trial}{scale}   = bubble_dims;
rawData(vp).f_coords{trial}{scale} = face_coords;
rp                 = add_alphaplane(patches{scale}, npic{1}, rawData(vp).b_centers{trial}{scale}, ...
                     rawData(vp).b_dims{trial}{scale}, rawData(vp).f_coords{trial}{scale}); %one patch size at a time
rp(rp > max_alpha) = max_alpha;
a_planes{scale}    = rp;
end


if rawData(vp).outmat(trial,17) > 4 % condition 'sad'
picNumber = rawData(vp).outmat(trial,3) + 60;
else
picNumber = rawData(vp).outmat(trial,3); % condition 'happy'
end

[t_stim, ~] = sum_to_target_ml(a_planes, npic(picNumber,:));
nPadded     = sprintf( '%04d', rawData(vp).outmat(trial, 1));
imwrite(t_stim/255, [rppath, conCodes{rawData(vp).outmat(trial, 17)}, ...
        '/comp-', rawData(vp).vpcode, '-', upper(rawData(vp).group(1)), ...
        '-', nPadded, '.png'], 'PNG');

clear t_stim
end
end
end


