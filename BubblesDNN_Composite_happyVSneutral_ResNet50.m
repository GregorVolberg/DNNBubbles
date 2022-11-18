% Bubbles Deep Neural Network Test
% https://de.mathworks.com/help/deeplearning/getting-started-with-deep-learning-toolbox.html

cOutputSize = [224 224 3]; % picture dimensions fpr AlexNet [227 227 3]
                           % picture dimensions for googleNet [224 224 3] 
                           % resnet101 [224 224 3]
                           

targetDirHC  = './dsetComposite/happyCorrect/';
targetDirHNC = './dsetComposite/sadCorrect/';

%     tmp = dir(targetDirHC);
%     fnames = {tmp(~[tmp.isdir]).name};
%     for k = 1:numel(fnames)
%     summedImage   = double(imread([targetDirHC, fnames{k}]));
%     conformImage  = imresize(summedImage, cOutputSize(1:2));
%     grayImage     = gray2ind(conformImage/255, 256);
%     faceImage     = cat(cOutputSize(3), grayImage, grayImage, grayImage);
%     imwrite(grayImage, [targetDirHC, fnames{k}]);
%     end
% 
%     tmp = dir(targetDirHNC);
%     fnames = {tmp(~[tmp.isdir]).name};
%     for k = 1:numel(fnames)
%     summedImage   = double(imread([targetDirHNC, fnames{k}]));
%     conformImage  = imresize(summedImage, cOutputSize(1:2));
%     grayImage     = gray2ind(conformImage/255, 256);
%     faceImage     = cat(cOutputSize(3), grayImage, grayImage, grayImage);
%     imwrite(faceImage, [targetDirHNC, fnames{k}]);
%     end


Numberofclasses = 2;
%https://de.mathworks.com/matlabcentral/answers/773617-errors-in-transfer-learning-using-resnet101
net = resnet50;
% 1. Convert DAGNetwork object to LayerGraph object
lgraph = layerGraph(net);   
% 2. Replace the last few layers
lgraph = replaceLayer(lgraph,'fc1000',...
  fullyConnectedLayer(Numberofclasses,'Name','fcNew'));
lgraph = replaceLayer(lgraph,'ClassificationLayer_fc1000',...
  classificationLayer('Name','ClassificationNew'));
% 4. Re-connect all the layers in the original order 
%    by using the support function createLgraphUsingConnections
layers = lgraph.Layers;
connections = lgraph.Connections;
lgraph = createLgraphUsingConnections(layers,connections);

% Train the network
miniBatchSize = 10;
%valFrequency = floor(numel(augimdsTrain.Files)/miniBatchSize);
options = trainingOptions('sgdm', ...
    'MiniBatchSize',miniBatchSize, ...
    'MaxEpochs',100, ...
    'InitialLearnRate',3e-4, ...
    'Shuffle','every-epoch', ...    
    'Verbose',false, ...
    'Plots','training-progress');

imdsTrain = imageDatastore({'./dsetComposite/happyCorrect/', './dsetComposite/happyNeutralCorrect/'}, ...
            'LabelSource', 'foldernames'); 

imdsTrain=augmentedImageDatastore(cOutputSize, imdsTrain,'ColorPreprocessing','rgb2gray');        
        
net = trainNetwork(imdsTrain,lgraph,options);








% imds = imageDatastore('./gpics', ...
%         'IncludeSubfolders', true, ...
%         'LabelSource', 'foldernames'); 
imdsTrain = imageDatastore({'./dsetComposite/happyCorrect/', './dsetComposite/happyNeutralCorrect/'}, ...
            'LabelSource', 'foldernames'); 

imdsValidation  = imageDatastore({'./pics/sad/', './pics/neutral/'}, ...
            'LabelSource', 'foldernames'); 
imdsValidation.Labels = categorical(cellstr(strvcat(repmat('happyCorrect', 30,1), repmat('happyNeutralCorrect', 30,1))));
%[imdsTrain,imdsValidation] = splitEachLabel(imds,0.7);


net = resnet101;%alexnet;
%label = classify(net, faceImages{1})
lgraph = layerGraph(net)    
% % https://de.mathworks.com/help/deeplearning/ug/train-deep-learning-network-to-classify-new-images.html
% 
% % inspect layers
%analyzeNetwork(net)
% 
% % replace two final layers
lgraph = layerGraph(net.Layers); %
[learnableLayer,classLayer] = findLayersToReplace(lgraph);
numClasses = numel(categories(imdsTrain.Labels));
if isa(learnableLayer,'nnet.cnn.layer.FullyConnectedLayer')
    newLearnableLayer = fullyConnectedLayer(numClasses, ...
        'Name','new_fc', ...
        'WeightLearnRateFactor',10, ...
        'BiasLearnRateFactor',10);
    
elseif isa(learnableLayer,'nnet.cnn.layer.Convolution2DLayer')
    newLearnableLayer = convolution2dLayer(1,numClasses, ...
        'Name','new_conv', ...
        'WeightLearnRateFactor',10, ...
        'BiasLearnRateFactor',10);
end

lgraph = replaceLayer(lgraph,learnableLayer.Name,newLearnableLayer);
newClassLayer = classificationLayer('Name','new_classoutput');
lgraph = replaceLayer(lgraph,classLayer.Name,newClassLayer);
%analyzeNetwork(lgraph);
% 
% % freeze weights of first 8 layers
layers = lgraph.Layers;
connections = lgraph.Connections;
for ii = 1:8
    props = properties(layers(ii));
    for p = 1:numel(props)
        propName = props{p};
        if ~isempty(regexp(propName, 'LearnRateFactor$', 'once'))
            layers(ii).(propName) = 0;
        end
    end
end
% 
% % reconnect all layers
lgraph = layerGraph();
for i = 1:numel(layers)
    lgraph = addLayers(lgraph,layers(i));
end
% 
for c = 1:size(connections,1)
    lgraph = connectLayers(lgraph,connections.Source{c},connections.Destination{c});
end
%analyzeNetwork(lgraph)
%
% train network
pixelRange = [-30 30];
scaleRange = [0.9 1.1];
imageAugmenter = imageDataAugmenter( ...
    'RandXReflection',true, ...
    'RandXTranslation',pixelRange, ...
    'RandYTranslation',pixelRange, ...
    'RandXScale',scaleRange, ...
    'RandYScale',scaleRange);
%augimdsTrain = augmentedImageDatastore(cOutputSize(1:2), imdsTrain, ...
%    'DataAugmentation',imageAugmenter);
augimdsTrain = augmentedImageDatastore(cOutputSize(1:2), imdsTrain);

%augimdsValidation = augmentedImageDatastore(cOutputSize(1:2),imdsValidation);

miniBatchSize = 10;
valFrequency = floor(numel(augimdsTrain.Files)/miniBatchSize);
options = trainingOptions('sgdm', ...
    'MiniBatchSize',miniBatchSize, ...
    'MaxEpochs',100, ...
    'InitialLearnRate',3e-4, ...
    'Shuffle','every-epoch', ...    
    'Verbose',false, ...
    'Plots','training-progress');

%    'ValidationData',augimdsValidation, ...
%    'ValidationFrequency',valFrequency, ...
% 
net = trainNetwork(augimdsTrain,lgraph,options);


[YPred,probs] = classify(net,imdsValidation);
accuracy = mean(YPred == imdsValidation.Labels);


%% visualize layer activity
% https://de.mathworks.com/help/deeplearning/ug/visualize-activations-of-a-convolutional-neural-network.html
im = imread('./pics/happy/image_01.png');
imshow(im);
imgSize = size(im);
imgSize = imgSize(1:2);

act1 = activations(net,im,'conv1');
sz = size(act1);
act1 = reshape(act1,[sz(1) sz(2) 1 sz(3)]);
actnorm = mat2gray(act1);
I = imtile(mat2gray(act1),'GridSize',[8 12]);
imshow(I)
featureIMG = zeros(size(actnorm, 1:2));
thresholdAt = 0.8;
for k = 1:size(actnorm,4)
featureIMG = featureIMG + squeeze(actnorm(:,:,1,k) >= thresholdAt);
end
%featureIMG = featureIMG./max(featureIMG(:));
fullfeatureIMG = imresize(featureIMG, imgSize);
Im = imtile({im,fullfeatureIMG});
imshow(Im);

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

 