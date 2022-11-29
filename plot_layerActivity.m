%% visualize layer activity
% https://de.mathworks.com/help/deeplearning/ug/visualize-activations-of-a-convolutional-neural-network.html
load('./data/trained_EmotionComposite_S30-NSSI_res50.mat', 'DNN');
%load('./data/trained_EmotionComposite_S31-control_res50.mat', 'DNN');



im = imread('./img/faces/female/img1.png');
im = repmat(im,1,1,3);
imshow(im);
imgSize = size(im);
imgSize = imgSize(1:2);

scaledIM = imresize(im, 'OutputSize', DNN.net.Layers(1).InputSize(1:2));

act1 = activations(DNN.net,scaledIM,'conv1');
%act1 = activations(DNN.net,scaledIM,'activation_49_relu');
%act1 = activations(DNN.net,scaledIM,'res5c_branch2c');
sz = size(act1);
act1 = reshape(act1,[sz(1) sz(2) 1 sz(3)]);
actnorm = mat2gray(act1);
I = imtile(mat2gray(act1),'GridSize',[8 8]);
imshow(I)

featureIMG = zeros(size(actnorm, 1:2));
thresholdAt = 0.7;
for k = 1:size(actnorm,4)
featureIMG = featureIMG + squeeze(actnorm(:,:,1,k) >= thresholdAt);
end
%featureIMG = featureIMG./max(featureIMG(:));

fullfeatureIMG = imresize(featureIMG, imgSize);
Im = imtile({im,fullfeatureIMG});
imshow(Im);

%alternativ: 
featIM = squeeze(mean(actnorm, 4));
fullfeatIM = imresize(featIM, imgSize);
imagesc(fullfeatIM); colormap(jet)

allLayers = {DNN.net.Layers.Name};
contains(allLayers, 'convol')


% act1ch62 = act1(:,:,:,62);
% act1ch62 = mat2gray(act1ch62);
% act1ch62 = imresize(act1ch62,imgSize);
% 
% I = imtile({im,act1ch62});
% imshow(I)

% stragety I: 
% - train DNN on full face images, happy-neutral and sad-neutral
% - use bubble-ized pictures as test stimuli

% strategy II
% - train DNN on bubbleized face images (as produced from patients and controls)
% - see how good full face images  as test stimuli are classified 
% - explore differences in layer activation 


% % strategies: construct learning stimuli from random bubbles and face
% % pictures

% bild zerlegen und so skalieren, dass einzelne SF sichtbar sind
grayBackground = zeros(size(bmp.struct_npic.npic{1}))+127;
scaleIntensity = 1; 
thresholdIntensity = 0.4; % remove anything below that value
% für jedes Teilbild
picNumber = 26;
figure;
for sfScale = 1:5 
maxAbsoluteValue = max(abs(bmp.struct_npic.npic{picNumber, sfScale}(:)));
scaleFactor = scaleIntensity/(maxAbsoluteValue/127);
scaledImage = bmp.struct_npic.npic{picNumber, sfScale} * scaleFactor + grayBackground;
belowThreshold = find((scaledImage > (127 - 127 * (thresholdIntensity/2))) & ...
                      (scaledImage < (127 + 127 * (thresholdIntensity/2))));
scaledImage(belowThreshold) = 127;
subplot(1, 5, sfScale);
image(scaledImage); colormap(gray); caxis([0 255]);
end

% hiervon kernelgrößen extrahieren für die verschiedenen Skalen (therohsl
% und maximale zahl der connected pixel
% dann wie in googlenet parallel kernel auf alle skalen

