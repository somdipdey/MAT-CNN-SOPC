%%Load pre-trained network model
net = vgg16;

%%Load DataSet
imds = imageDatastore('/Users/somdipdey/Downloads/Motorway-Traffic-Categorised', ...
    'IncludeSubfolders',true, ...
    'LabelSource','foldernames');

%Split DataSet into two sets: Training & Validation. Here Split is done on
%9:1 ratio.
[imdsTrain,imdsValidation] = splitEachLabel(imds,0.8,'randomized');

%Select 9 random image numbers to show
numTrainImages = numel(imdsTrain.Labels);
idx = randperm(numTrainImages,9);
figure

%Show images of 9 randoms
for i = 1:9
    subplot(3,3,i)
    I = readimage(imdsTrain,idx(i));
    imshow(I);
    title(char(imdsTrain.Labels(idx(i))));
end

%Get inputSize of images
inputSize = net.Layers(1).InputSize;

%Get layers to transfer
layersTransfer = net.Layers(1:end-3);

%Fetch num of classes
numClasses = numel(categories(imdsTrain.Labels));

%Add a fully connected layer, a dropout layer, a softmax layer
layers = [
    layersTransfer
    fullyConnectedLayer(numClasses,'WeightLearnRateFactor',20,'BiasLearnRateFactor',20)
    dropoutLayer(0.5,'Name','DROPOUT')
    softmaxLayer
    classificationLayer];

%%Train network
% Augment dataset to prevent network from overfitting and memorizing
% training images
pixelRange = [-30 30];
imageAugmenter = imageDataAugmenter( ...
    'RandXReflection',true, ...
    'RandXTranslation',pixelRange, ...
    'RandYTranslation',pixelRange);

%Augment training dataset images along with resizing
augimdsTrain = augmentedImageDatastore(inputSize(1:2),imdsTrain, ...
    'DataAugmentation',imageAugmenter);

%Just resize the validation dataset
augimdsValidation = augmentedImageDatastore(inputSize(1:2),imdsValidation);

%Specify training options
options = trainingOptions('sgdm', ...
    'MiniBatchSize',10, ...
    'MaxEpochs',10, ...
    'InitialLearnRate',1e-4, ...
    'ValidationData',augimdsValidation, ...
    'ValidationFrequency',3, ...
    'ValidationPatience',Inf, ...
    'Verbose',false, ...
    'ExecutionEnvironment','auto', ...
    'Plots','training-progress');

%Finally start training
netTransfer = trainNetwork(augimdsTrain,layers,options);

%%Classify Validation Images
[YPred,scores] = classify(netTransfer,augimdsValidation);

%%Display four sample validation images with their predicted labels
idx = randperm(numel(imdsValidation.Files),9);
figure
for i = 1:9
    subplot(3,3,i)
    I = readimage(imdsValidation,idx(i));
    imshow(I)
    label = YPred(idx(i));
    title(string(label));
end

%%Calculate the classification accuracy on the validation set. Accuracy is the fraction of labels that the network predicts correctly
YValidation = imdsValidation.Labels;
accuracy = mean(YPred == YValidation);