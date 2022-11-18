% =======================================
% Extracts still pictures (*.png) from mp4.
% Uses the dynamic Karolinska Directed Emotional Faces dataset (KDEF-dyn)
% https://www.kdef.se/versions.html
% Calvo et al. (2018), Sci Reports, DOI:10.1038/s41598-018-35259-w
%
%========================================

function [] = make_pngs_DNN()
filepath = 'D:\BHK_Bubbles\KDEFdynI\I. KDEF-dyn I\S4 Stimuli (Video-clips)';
destpath = 'D:\BubblesDNN2\gpics\';
PyramidToolsPath = 'C:\Users\LocalAdmin\Documents\m-lib\matlabPyrTools-master';
SHINEpath        = 'C:\Users\LocalAdmin\Documents\m-lib\SHINEtoolbox';
addpath(genpath(PyramidToolsPath), SHINEpath);

minIntensity = 20;  % for calculating face mask; so that it includes pixels with average intensity of minIntensity
erode1       = 18;  % computes rim for smooting face oval, layer 1; rim size is erode1/2 = 0:9 = 10 Pixel
erode2       = 38;  % computes rim for smooting face oval, layer 2; rim size is 10 Pixel
largePicSize = 700; % uses larger image in intermediate processing (border detection for erosion)
targetLum    = [127 35]; % target luminance, for luminance matching (mean and sd)
smoothkernel = 3;
do_reframe   = 1;   % add gray frame around final face pic
framesize    = 1.4;   % factor, relative to face size
do_resample  = 1;   % in order to achieve comparable SF scales as Butler Fiset 2017
fraction     = 0.75; %
plotflag     = 0; % plot SF scales

cOutputSize = [227 227 3]; % picture dimensions fpr AlexNet [227 227 3]
                           % picture dimensions for googleNet [224 224 3]   


[pathname, foldernames, filenames] = getNames(filepath); % see subfunction
foldernames([2:3, 5:6]) = [];
filenames([2:3, 5:6]) = []; % 
conditions = {'happy', 'sad'};


%tmp  = readtable([destpath, 'Zeitstempel_SAD_Bubbles_low.xlsx']);
%t    = round(tmp{:,6});
%code = tmp{:,1};
%[t, code] = textread('sadtimes.txt', '%f %3c');
%t = round(t);
targetFrames = [6:1:9];

idx=0;
for k = 1:numel(foldernames)
    mkdir([destpath, conditions{k}]);
    for m = 1:numel(filenames{k})
        for i = 1:numel(targetFrames)
            idx = idx+1;
            v = VideoReader([pathname, '\', foldernames{k}, '\', filenames{k}{m}]);
            frame{idx} = double(rgb2gray(read(v, targetFrames(i))));
            frameName{idx} = ['./', conditions{k}, '/',v.Name(1:(end-4)), '-Frame', ...
                sprintf('%02s', num2str(targetFrames(i))), '.png'];
            end
        end
end

[graymask, ~] = getFaceMask(frame, minIntensity); % get face mask
[largePics, rim] = applyFaceMask(frame, graymask, largePicSize, erode1, erode2); % apply mask and scale up
[smoothedPics, picdims] = matchLuminanceAndSmooth(largePics, rim, targetLum, smoothkernel); % match luminance, smooth edges and scale down

% optional: resample prior to application of Laplace pyramid
% as in experiment
if do_resample
    [smoothedPics, picdims] = reSamp(smoothedPics, fraction); 
end
facedims = picdims;

% optional: reframe (ie add frame)
% as in experiment
if do_reframe
    [smoothedPics, picdims, margins] = reFrame(smoothedPics, framesize);
end

for img = 1:numel(smoothedPics)
    conformImage = imresize(smoothedPics{img}, cOutputSize(1:2));
    grayImage    = gray2ind(conformImage/255, 256);
    faceImage    = cat(cOutputSize(3), grayImage, grayImage, grayImage);
    imwrite(faceImage, frameName{img});
end

end



