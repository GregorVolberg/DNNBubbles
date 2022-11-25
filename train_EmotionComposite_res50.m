function [] = train_EmotionComposite_res50()
% Bubbles Deep Neural Network Test
% https://de.mathworks.com/help/deeplearning/getting-started-with-deep-learning-toolbox.html
tic
addpath('./func');

% give image folders
imds = imageDatastore({'./img/EmotionComposite/happyCorrect',...
                       './img/EmotionComposite/happyNeutralCorrect'}); 
Labels = cell(numel(imds.Files), 1);
Labels(contains(imds.Files, '/happyCorrect'))   = {'happy'};
Labels(contains(imds.Files, '/happyNeutralCorrect')) = {'neutral'};
imds.Labels = categorical(Labels);

[imdsTrain,imdsValidation] = splitEachLabel(imds,0.7);
numClasses = numel(categories(imdsTrain.Labels));

% load net architecture
load('./data/resnet50.mat', 'net'); % pretrained
inputSize = net.Layers(1).InputSize; 

% convert to layer graph in order to modify layers
if isa(net,'SeriesNetwork') 
  lgraph = layerGraph(net.Layers); 
else
  lgraph = layerGraph(net);
end 

% find and replace classification layer and preceding learnable layer
[learnableLayer, classLayer] = findLayersToReplace(lgraph);

% class(learnableLayer)
newLearnableLayer = fullyConnectedLayer(numClasses, ...
        'Name','new_fc', ...
        'WeightLearnRateFactor',10, ...
        'BiasLearnRateFactor',10);
lgraph = replaceLayer(lgraph,learnableLayer.Name,newLearnableLayer);

newClassLayer = classificationLayer('Name','new_classoutput');
lgraph = replaceLayer(lgraph,classLayer.Name,newClassLayer);

% freeze first layers and re-connect
layers = lgraph.Layers;
connections = lgraph.Connections;
layers(1:10) = freezeWeights(layers(1:10));
lgraph = createLgraphUsingConnections(layers,connections);

% image augmentation
pixelRange = [-10 10];
scaleRange = [0.9 1.1];
imageAugmenter = imageDataAugmenter( ...
    'RandXReflection',true, ...
    'RandXTranslation',pixelRange, ...
    'RandYTranslation',pixelRange, ...
    'RandXScale',scaleRange, ...
    'RandYScale',scaleRange);

augimdsTrain = augmentedImageDatastore(inputSize(1:2), imdsTrain, ...
    'DataAugmentation',imageAugmenter,'ColorPreprocessing', 'gray2rgb');
augimdsValidation = augmentedImageDatastore(inputSize(1:2),imdsValidation,...
    'ColorPreprocessing', 'gray2rgb');

% training options
miniBatchSize = 6;
valFrequency  = floor(numel(augimdsTrain.Files)/miniBatchSize);
options = trainingOptions('sgdm', ...
    'MiniBatchSize',miniBatchSize, ...
    'MaxEpochs',100, ...
    'InitialLearnRate',3e-4, ...
    'Shuffle','every-epoch', ...
    'ValidationData',augimdsValidation, ...
    'ValidationFrequency',valFrequency, ...
    'Verbose',false, ...
    'Plots','none');

% train and predict 
net = trainNetwork(augimdsTrain,lgraph,options);
[YPred, ~] = classify(net,augimdsValidation);
accuracy = mean(YPred == imdsValidation.Labels);

callingFunction = dbstack;
stopTime        = toc;

DNN.minutes  = floor(stopTime/60);
DNN.call     = callingFunction.name;
DNN.folders  = imds.Folders;
DNN.labels   = unique(imds.Labels);
DNN.accuracy = accuracy;
DNN.net      = net;
save('./data/trained_EmotionComposite_res50.mat', 'DNN', '-v7.3');

fprintf('\nMean accuracy: %.2f', accuracy);
fprintf('\nFinished in %i:%i minutes.\n', floor(stopTime/60), round(mod(stopTime,60)));
end
 