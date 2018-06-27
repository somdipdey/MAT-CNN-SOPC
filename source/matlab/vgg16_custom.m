function vgg16_custom()
    %%Load pre-trained network model
    net = vgg16;

    %%Load DataSet
    imds = imageDatastore('/Users/somdipdey/Downloads/Motorway-Traffic-Categorised', ...
        'IncludeSubfolders',true, ...
        'LabelSource','foldernames');

    %Split DataSet into two sets: Training & Validation. Here Split is done on
    %9:1 ratio.
    [imdsTrain,imdsValidation] = splitEachLabel(imds,0.9,'randomized');

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
    %custom built on 4 categories -->
    %last_layer = 39;
    %last_layer_name = net.Layers(last_layer).Name;
    %visual_channels = 1:6;
    %thisI = deepDreamImage(net,last_layer,visual_channels, ...
    %    'PyramidLevels',1, ...
    %    'NumIterations',28);
    %figure;
    %montage(thisI);
    %title(['Layer: ',last_layer_name,' Features'])
    
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
        'ExecutionEnvironment','cpu', ...
        'Plots','training-progress');

    %Finally start training
    netTransfer = trainNetwork(augimdsTrain,layers,options);
    
    %%Classify Validation Images
    [YPred,scores] = classify(netTransfer,augimdsValidation);
    %Display Predication and it's score
    fprintf('Prediction: %s' , YPred , ', Scores: ');
    disp(scores);
    fprintf('\n');

    %%Display nine sample validation images with their predicted labels
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
    fprintf('Accuracy: %s' , accuracy);
    
    %Test classification on a completely different set
    %Just resize the validation dataset
        %%Load DataSet
    test_imds = imageDatastore('/Users/somdipdey/Documents/MATLAB/Add-Ons/Collections/Deep Learning Tutorial Series/code/AHS/test_dataset', ...
        'IncludeSubfolders',true, ...
        'LabelSource','foldernames');
    test_augimdsValidation = augmentedImageDatastore(inputSize(1:2),test_imds);
    %%Classify Test Images
    [YPred2,scores2] = classify(netTransfer,test_augimdsValidation);
    %Display Predication and it's score
    fprintf('\n Testing on a completely new dataset \n');
    fprintf('Prediction: %s' , YPred2 , ', Scores: ');
    disp(scores2);
    fprintf('\n');
    %%Calculate the classification accuracy on the validation set. Accuracy is the fraction of labels that the network predicts correctly
    test_YValidation = test_imds.Labels;
    accuracy2 = mean(YPred2 == test_YValidation);
    %Display Accuracy
    fprintf('\n Accuracy: %s' , accuracy2);
    
    %%Display nine sample validation images with their predicted labels
    idx2 = randperm(numel(test_imds.Files),9);
    figure
    for i = 1:9
        subplot(3,3,i)
        I = readimage(test_imds,idx2(i));
        imshow(I)
        label = YPred2(idx2(i));
        title(string(label));
    end
end