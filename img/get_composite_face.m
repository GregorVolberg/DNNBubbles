% machine learning cooperation with Jens, Philipp, Alex
% uses BubbblesRawData_zwisch
vp_selection = {'S31', 'S32', 'S35', 'S36', ...
                'S37', 'S41', 'S45', 'S46'};

addpath('./func');
outFileName = 'BubblesProtocolComposite.txt';
rppath = './dsetComposite/';
conCodes = {'happyCorrect/', 'happyIncorrect/', 'happyNeutralCorrect/', 'happyNeutralIncorrect/', ...
            'sadCorrect/', 'sadIncorrect/', 'sadNeutralCorrect/', 'sadNeutralIncorrect/'};

tfolders = dir('./dsetComposite/');
if ~all(ismember(conCodes, strcat({tfolders.name}, '/')))
for k = 1:numel(conCodes)
    mkdir(['./dsetComposite/' ,conCodes{k}]);    
end
end

load ('../raw/BubblesRawData_zwisch.mat', 'rawData');
picfilename = ['..\..\', rawData(1).stmfile]; % 'p5_struct_npic_470x349.mat'
[~, ~, npic, mids] = load_stimuli(picfilename);
%patches  = get_patches(rawData(1).facedims, mids, rawData(1).num_cycles, rawData(1).sd_Gauss);
patches = get_patches_ml(rawData(1).facedims, mids, rawData(1).num_cycles, rawData(1).sd_Gauss, 4);

get_composite_ml(rawData, rppath, vp_selection, patches, npic, outFileName, conCodes); % write to disk


% 
% facepath = [rppath, 'faces/'];
% if ~ismember('faces', {tfolders.name})
% mkdir(facepath);
% end
% 
% select unique pics from 120 npics
% tmp = load(picfilename);
% faceNames = tmp.struct_npic.names; clear tmp
% for k = 1:numel(faceNames)
% woPath{k} = faceNames{k}((end-14):(end));
% end
% 
% [uniqueFaces, indx1, indx2] = unique(woPath, 'stable');
% 
% for pic = 1 : size(npic, 1)
% for scale = 1 : size(npic, 2)
% picText = ['f', num2str(pic  + 1000 * scale)];
% face = npic{pic, scale} + 127;
% save([facepath, picText, '.mat'], 'face'); 
% imwrite(face/255, [facepath, picText, '.png'], 'PNG')
% end
% end
% 

% 1: NE-HA happy + correct 
% 2: NE-HA happy + incorrect
% 3: NE-HA neutral + correct
% 4: NE-HA neutral + incorrect
% 5: NE-SA sad + correct 
% 6: NE-SA sad + incorrect
% 7: NE-SA neutral + correct
% 8: NE-SA neutral + incorrect

% eg = {'S14', 'S15', 'S16', 'S19', 'S24', 'S23', 'S17', ...
%       'S30', 'S20', 'S33', 'S34', 'S38', 'S39', 'S40', ...
%       'S42', 'S44', 'S56'}; % experimental group nr13?
% cg = {'S07', 'S08', 'S10', 'S12', 'S18', 'S22', 'S21', ...
%       'S25', 'S26', 'S27', 'S28', 'S29', 'S31', 'S32', ...
%       'S35', 'S36', 'S37', 'S41', 'S45', 'S46', 'S47', ...
%       'S48', 'S49', 'S50', 'S51', 'S52', 'S53', 'S54', 'S56', 'S57'};% control group
