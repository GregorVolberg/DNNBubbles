% ==================================
% Creates Laplacian Pyramids from *.png files as created with 'make_png.m'
% 
% Runs with R2016b or newer
% GV 10.02.2020
% ==================================

function [struct_npic] = make_npics()

% set paths
PyramidToolsPath = 'C:\Users\Gregor\Documents\m-lib\matlabPyrTools-master';
SHINEpath        = 'C:\Users\Gregor\Documents\m-lib\SHINEtoolbox';
addpath(genpath(PyramidToolsPath), SHINEpath);

% picture directories
facedirs   =  {'C:\Users\Gregor\Filr\Meine Dateien\Bubbles_BKH\KDEFdynI\I. KDEF-dyn I\S6_Bitmaps\1_Neutral-Happiness\';
               'C:\Users\Gregor\Filr\Meine Dateien\Bubbles_BKH\KDEFdynI\I. KDEF-dyn I\S6_Bitmaps\4_Neutral-Sadness\'};

% resulting file name
savefilename = 'struct_npic_762x498.mat';
           
% flags and constants 
nscales      = 6;
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


[pics, picnames] = readpics(facedirs, '*.png'); % read pictures
[graymask, avgpic] = getFaceMask(pics, minIntensity); % get face mask
[largePics, rim] = applyFaceMask(pics, graymask, largePicSize, erode1, erode2); % apply mask and scale up
[smoothedPics, picdims] = matchLuminanceAndSmooth(largePics, rim, targetLum, smoothkernel); % match luminance, smooth edges and scale down

% optional: resample prior to application of Laplace pyramid
if do_resample
    [smoothedPics, picdims] = reSamp(smoothedPics, fraction); 
end
facedims = picdims;

% optional: reframe (ie add frame)
if do_reframe
    [smoothedPics, picdims, margins] = reFrame(smoothedPics, framesize);
end

[scales, npic, ~] = getScales(smoothedPics, nscales, plotflag); % compute scales; last argument is plotting flag
scales = scales/framesize; % because it is per face, framesize is face size
mids = diff(scales)/2 + scales(1:(end-1)); % verify scales by visual inspection; need to double in each step

lp = npic(:,6);
npic_matrix = reshape([lp{:}], size(lp{1},1), size(lp{1},2), numel(lp));
%common_avg  = squeeze(mean(npic_matrix(:, :, :), 3));
NE_HA_avg   = squeeze(mean(npic_matrix(:, :, 1:80), 3));
NE_SA_avg   = squeeze(mean(npic_matrix(:, :, 81:160), 3));

% save
struct_npic.npic     = npic; 
struct_npic.names    = picnames';
struct_npic.picdims  = picdims;
struct_npic.facedims = facedims;
%struct_npic.Factor   = framesize;
struct_npic.margins  = margins;
struct_npic.scales   = scales;
struct_npic.mids     = mids;
struct_npic.NE_HA_lpavg = NE_HA_avg;
struct_npic.NE_SA_lpavg = NE_SA_avg;
struct_npic.date     = date();
save(['struct_npic_', num2str(picdims(1)), 'x', num2str(picdims(2))], 'struct_npic');
end

