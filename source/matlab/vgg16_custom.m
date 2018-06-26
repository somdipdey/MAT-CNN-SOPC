function vgg16_custom()
    %%Load pre-trained network model
    net = vgg16;

    %%Load DataSet
    imds = imageDatastore('/Users/somdipdey/Downloads/Motorway-Traffic-Categorised', ...
        'IncludeSubfolders',true, ...
        'LabelSource','foldernames');

    %Split DataSet into two sets: Training & Validation. Here Split is done on
    %9:1 ratio.
    [imdsTrain,imdsValidation] = splitEachLabel(imds,0.7,'randomized');

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
        fullyConnectedLayer(numClasses,'WeightLearnRateFactor',20,'BiasLearnRateFactor',20,'Name','lastFCCustom')
        dropoutLayer(0.5,'Name','DROPOUT')
        softmaxLayer
        classificationLayer];

    %Display extracted feature of the last fully connected layer, which is
    %custom built on 4 categories
    last_layer = 39;
    last_layer_name = net.Layers(last_layer).Name;
    visual_channels = 1:6;
    thisI = deepDreamImage(net,last_layer,visual_channels, ...
        'PyramidLevels',1, ...
        'NumIterations',28);
    figure;
    montage(thisI);
    title(['Layer: ',last_layer_name,' Features'])
    
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
    %Display Predication and it's score
    disp('Prediction: ' + YPred + ', Scores: ' + scores);

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
    %Display Accuracy
    disp('Accuracy: ' + accuracy);
end