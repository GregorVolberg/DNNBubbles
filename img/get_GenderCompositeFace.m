% generate bubbleized faces for gender classification task
function [] = get_GenderCompositeFace()
addpath('../func', genpath('~/m-lib/Psychtoolbox'));

vpSelection       = {'S01', 'S03', 'S05', 'S06'};
ProtocolFileName  = 'BubblesGenderComposite.txt';
targetDir         = './GenderComposite/';
conditionCodes    = {'female', 'male'};
dataDir           = '../data/';
rawDataFile       = [dataDir, 'BubblesFacesRaw.mat'];

for k = 1:numel(conditionCodes)
    [~, ~] = mkdir([targetDir ,conditionCodes{k}, '/correct']);
    [~, ~] = mkdir([targetDir ,conditionCodes{k}, '/incorrect']);
end

load (rawDataFile, 'rawData');
[~, ~, npic, mids] = load_stimuli([dataDir, rawData(1).stmfile]);
patches = get_patches_ml(rawData(1).facedims, mids, rawData(1).num_cycles, rawData(1).sd_Gauss, 4);
get_composite_gender_ml(rawData, targetDir, vpSelection, patches, npic, ProtocolFileName, [conditionCodes, '/']); % write to disk

end