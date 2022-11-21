function [] = get_EmotionCompositeFace()

addpath('../func', genpath('~/m-lib/Psychtoolbox'));

vp_selection = {'S31', 'S32', 'S35', ...
                'S30', 'S33', 'S34'};

targetDir         = './EmotionComposite/';
conditionCodes    = {'happyCorrect/', 'happyIncorrect/', 'happyNeutralCorrect/', 'happyNeutralIncorrect/', ...
                  'sadCorrect/', 'sadIncorrect/', 'sadNeutralCorrect/', 'sadNeutralIncorrect/'};
dataDir           = '../data/';
rawDataFile       = [dataDir, 'BubblesRawData_zwisch.mat'];

for k = 1:numel(conditionCodes)
    [~, ~] = mkdir([targetDir ,conditionCodes{k}]);
end

load (rawDataFile, 'rawData');
[~, ~, npic, mids] = load_stimuli([dataDir, rawData(1).stmfile, '.mat']);
patches = get_patches_ml(rawData(1).facedims, mids, rawData(1).num_cycles, rawData(1).sd_Gauss, 4);
get_composite_emotion_ml(rawData, targetDir, vp_selection, patches, npic, conditionCodes); % write to disk

end