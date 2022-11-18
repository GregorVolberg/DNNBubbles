% Bubbles Deep Neural Network Test
% https://de.mathworks.com/help/deeplearning/getting-started-with-deep-learning-toolbox.html

bmp         = load('D:\BHK_Bubbles\p5_struct_npic_470x349.mat');
cOutputSize = [227 227 3]; % picture dimensions fpr AlexNet [227 227 3]
                           % picture dimensions for googleNet [224 224 3]   
[faceImages, conditionCode] = getFaceImages(bmp, cOutputSize); % 1 = happy emotion
                                                  % 2 = happy neutral
                                                  % 3 = sad emotion
                                                  % 4 = sad neutral
                                                  

% better save as bitmap for training
for k = 1:max(find(conditionCode==3)) % leave out #4
    switch conditionCode(k)
        case 1 
            folderName = 'happy';
        case 2
            folderName = 'neutral';
        case 3
            folderName = 'sad';
    end
    
imageName = ['.\pics\', folderName, '\image_', sprintf('%02s', num2str(k)), '.png'];
imwrite(faceImages{k}, imageName);
end


imds = imageDatastore('pics', ...
        'IncludeSubfolders', true, ...
        'LabelSource', 'foldernames'); 
[imdsTrain,imdsValidation] = splitEachLabel(imds,0.7);

save('faceImages.mat', 'faceImages');
save('conditionCode.mat', 'conditionCode');

% % % example images
% % figure; % pics F02 und M10; (also M24 M25 M31 M34)
% % subplot(2, 3, 1); imagesc(faceImages{13}); colormap(gray); title('happy');
% % subplot(2, 3, 2); imagesc(faceImages{42}); colormap(gray); title('neutral');
% % subplot(2, 3, 3); imagesc(faceImages{69}); colormap(gray); title('sad');
% % subplot(2, 3, 4); imagesc(faceImages{26}); colormap(gray);
% % subplot(2, 3, 5); imagesc(faceImages{51}); colormap(gray);
% % subplot(2, 3, 6); imagesc(faceImages{75}); colormap(gray);
% 

net = alexnet;
%label = classify(net, faceImages{1})
    
% % https://de.mathworks.com/help/deeplearning/ug/train-deep-learning-network-to-classify-new-images.html
% 
% % inspect layers
analyzeNetwork(net)
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
analyzeNetwork(lgraph);
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
analyzeNetwork(lgraph)
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
augimdsTrain = augmentedImageDatastore(cOutputSize(1:2), imdsTrain, ...
    'DataAugmentation',imageAugmenter);

augimdsValidation = augmentedImageDatastore(cOutputSize(1:2),imdsValidation);

miniBatchSize = 10;
valFrequency = floor(numel(augimdsTrain.Files)/miniBatchSize);
options = trainingOptions('sgdm', ...
    'MiniBatchSize',miniBatchSize, ...
    'MaxEpochs',100, ...
    'InitialLearnRate',3e-4, ...
    'Shuffle','every-epoch', ...
    'ValidationData',augimdsValidation, ...
    'ValidationFrequency',valFrequency, ...
    'Verbose',false, ...
    'Plots','training-progress');
% 
net = trainNetwork(augimdsTrain,lgraph,options);
 
[YPred,probs] = classify(net,augimdsValidation);
accuracy = mean(YPred == imdsValidation.Labels);

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

%% ======== subfunctions
function [faceImages, conditionCode] = getFaceImages(bmp, outputSize)
cImageSize      = size(bmp.struct_npic.npic{1});
cNumberOfImages = size(bmp.struct_npic.npic, 1);
cNumberOfScales = size(bmp.struct_npic.npic, 2);
faceImages      = cell(1, cNumberOfImages);

    for j = 1:cNumberOfImages
    allScaleImage = [bmp.struct_npic.npic{j, 1:6}];
    singleImage   = reshape(allScaleImage, [cImageSize, cNumberOfScales]);
    summedImage   = squeeze(sum(singleImage, 3));
    conformImage  = imresize(summedImage, outputSize(1:2));
    grayImage     = gray2ind(conformImage/255, 256);
    faceImages{j} = cat(outputSize(3), grayImage, grayImage, grayImage);
    end
    
conditionCode = [ones(1,30), ones(1,30) + 1, ones(1,30 )+ 2, ones(1,30) + 3];
end
 